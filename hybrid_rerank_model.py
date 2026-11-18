"""Frozen-generator, token-CE rerankers with exactly matched trainable capacity.

CONTROL: local frozen evidence -> sentence BiGRU/Transformer; token-only queries.
HYBRID: additionally conditions sentence context on the frozen candidate distribution,
        and uses candidate-specific queries against all sentence positions.
Neither arm consumes targets, ranks, sentence losses or margin objectives. Candidate
order is not a feature. Checkpoints contain the full model (including frozen base).
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

import context_pilot as cp
import evaluate as ev
import pilot_arch as pa

ARM_PREFIX = 'hybrid_rerank'
CONFIGS = ('CONTROL', 'HYBRID')
DEFAULT_CONFIG = dict(dim=256, heads=8, layers=3, feedforward=1024,
                      dropout=0.1, char_embedding=64, query_chunk=256)


def arm_name(config_name):
    return f'{ARM_PREFIX}_{config_name}'


class FullCandidateEncoder(nn.Module):
    """Full, packed bidirectional character encoding, no character truncation."""
    def __init__(self, nchar, cfg):
        super().__init__()
        d = cfg['dim']
        self.embedding = nn.Embedding(nchar, cfg['char_embedding'], padding_idx=0)
        self.gru = nn.GRU(cfg['char_embedding'], d // 2, batch_first=True,
                          bidirectional=True)
        self.out = nn.Linear(d, d)

    def forward(self, ids, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.out(torch.cat((h[-2], h[-1]), dim=-1))


def _candidate_tables(payload):
    # An independent lossless alphabet is essential: the stripped-input alphabet
    # in some generator checkpoints does not contain every accented character.
    forms, cand, c = payload['forms'], payload['cand'], payload['max_c']
    words = sorted({word for stem in forms for word in cand[stem]})
    alphabet = {ch: i + 1 for i, ch in enumerate(sorted(set(''.join(words))))}
    word_id = {word: i for i, word in enumerate(words)}
    width = max([1] + [len(word) for word in words])
    ids = torch.zeros(len(words), width, dtype=torch.long)
    lengths = torch.ones(len(words), dtype=torch.long)
    index = torch.full((len(forms), c), -1, dtype=torch.long)
    for i, word in enumerate(words):
        ids[i, :len(word)] = torch.tensor([alphabet[ch] for ch in word], dtype=torch.long)
        lengths[i] = max(1, len(word))
    for i, stem in enumerate(forms):
        if not 0 < len(cand[stem]) <= c:
            raise ValueError('Candidate count is outside checkpoint max_c')
        index[i, :len(cand[stem])] = torch.tensor([word_id[w] for w in cand[stem]])
    return ids, lengths, index, len(alphabet) + 1


class HybridReranker(nn.Module):
    def __init__(self, payload, config_name='HYBRID', config=None, load_base=True):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config is not None:
            unknown = set(config) - set(cfg)
            if unknown:
                raise ValueError(f'Unknown reranker configuration keys: {sorted(unknown)}')
            cfg.update(config)
        if config_name not in CONFIGS:
            raise ValueError(f'Unknown configuration {config_name}')
        d = cfg['dim']
        if (d < 2 or d % 2 or cfg['heads'] < 1 or d % cfg['heads'] or
                cfg['layers'] < 1 or cfg['query_chunk'] < 1):
            raise ValueError('Require even dim divisible by heads and positive layers/query_chunk')
        if not payload['config'].get('packed_chars'):
            raise ValueError('Requires a packed-character frozen factorized generator')
        self.cfg, self.config_name = cfg, config_name
        self.max_c = payload['max_c']
        self.base = pa.FactorizedDiac(
            payload['nchar'], self.max_c, len(payload['signatures']['modifier']),
            len(payload['signatures']['tone']), **payload['config'])
        if load_base:
            self.base.load_state_dict(payload['model'], strict=True)
        self.base.requires_grad_(False).eval()
        self.register_buffer('tone_table', payload['tone_table'].clone())
        self.register_buffer('mod_table', payload['mod_table'].clone())
        ids, lengths, index, nchar = _candidate_tables(payload)
        self.register_buffer('char_ids', ids)
        self.register_buffer('char_len', lengths)
        self.register_buffer('candidate_index', index)
        self.candidate = FullCandidateEncoder(nchar, cfg)
        self.hidden_projection = nn.Linear(payload['config']['dim'], d)
        self.context_gru = nn.GRU(d, d // 2, batch_first=True, bidirectional=True)
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cfg['heads'], cfg['feedforward'],
                                       cfg['dropout'], batch_first=True),
            cfg['layers'], enable_nested_tensor=False)
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d)
        self.interaction_out = nn.Linear(d, d)
        self.scorer = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1, bias=False))
        nn.init.zeros_(self.scorer[-1].weight)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    frozen=sum(p.numel() for p in self.parameters() if not p.requires_grad))

    @staticmethod
    def _validate_mask(mask):
        lengths = mask.sum(-1)
        if not torch.equal(mask, torch.arange(mask.shape[1], device=mask.device)[None]
                           < lengths[:, None]):
            raise ValueError('Sentence masks must be right-padded contiguous prefixes')
        return lengths

    def frozen(self, xc, cl, f, ln, mask):
        """Frozen hidden/scores; omit empty sentences to avoid all-masked attention."""
        self.base.eval()
        b, t = f.shape
        hidden = self.hidden_projection.weight.new_zeros((b, t, self.base.norm.normalized_shape[0]))
        scores = hidden.new_full((b, t, self.max_c), ev.NEG)
        rows = mask.any(-1)
        with torch.no_grad():
            if rows.any() and t:
                h = cp.AttnDiac.forward(self.base, xc[rows], cl[rows], mask=mask[rows])
                safe_f = f[rows].masked_fill(~mask[rows], 0)
                s = (self.base.tone_head(h).gather(-1, self.tone_table[safe_f]) +
                     self.base.mod_head(h).gather(-1, self.mod_table[safe_f]))
                hidden[rows] = h.masked_fill(~mask[rows, :, None], 0)
                scores[rows] = s
            valid = mask[..., None] & (torch.arange(self.max_c, device=f.device) < ln[..., None])
        return hidden.detach(), scores.masked_fill(~valid, ev.NEG).detach()

    def candidate_embeddings(self, f, ln, mask):
        """Encode each unique live word once; return live indices and embeddings only."""
        safe_f = f.masked_fill(~mask, 0)
        index = self.candidate_index[safe_f]
        valid = mask[..., None] & (torch.arange(self.max_c, device=f.device) < ln[..., None])
        positions = valid.nonzero(as_tuple=False)
        chosen = index[valid]
        if chosen.numel() and (chosen < 0).any():
            raise ValueError('Live candidate count exceeds dictionary candidates')
        unique, inverse = chosen.unique(sorted=True, return_inverse=True)
        if not unique.numel():
            return positions, self.hidden_projection.weight.new_empty((0, self.cfg['dim']))
        return positions, self.candidate(self.char_ids[unique], self.char_len[unique])[inverse]

    def _positions(self, length, reference):
        d = self.cfg['dim']
        pos = torch.arange(length, device=reference.device, dtype=torch.float32)[:, None]
        freq = torch.exp(torch.arange(0, d, 2, device=reference.device).float()
                         * (-math.log(10000.0) / d))
        phase = pos * freq
        return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2).to(reference.dtype)

    def logits(self, xc, cl, f, ln, mask):
        mask = mask.bool()
        lengths = self._validate_mask(mask)
        if ((ln < 0) | (ln > self.max_c))[mask].any():
            raise ValueError('Live candidate counts outside [0, max_c]')
        hidden, base_scores = self.frozen(xc, cl, f, ln, mask)
        positions, cand = self.candidate_embeddings(f, ln, mask)
        if not positions.numel():
            # Keep backward legal even for an entirely empty training batch.
            return base_scores + self.scorer[-1].weight.sum() * 0
        bidx, tidx, cidx = positions.unbind(-1)
        local = torch.tanh(self.hidden_projection(hidden)).masked_fill(~mask[..., None], 0)
        context_input = local
        if self.config_name == 'HYBRID':
            # Detached generator probabilities, NEVER targets or current corrections.
            valid = mask[..., None] & (torch.arange(self.max_c, device=f.device) < ln[..., None])
            distribution = base_scores.masked_fill(~valid, -torch.inf)
            distribution = distribution.masked_fill(~valid.any(-1, keepdim=True), 0)
            probability = distribution.softmax(-1).masked_fill(~valid, 0).detach()
            flat_index = bidx * f.shape[1] + tidx
            expected = local.new_zeros((f.numel(), self.cfg['dim'])).index_add(
                0, flat_index, probability[bidx, tidx, cidx, None] * cand).view_as(local)
            context_input = local + expected
        rows = lengths > 0
        packed = nn.utils.rnn.pack_padded_sequence(
            context_input[rows], lengths[rows].cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.context_gru(packed)
        sentence, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=f.shape[1])
        sentence = sentence + self._positions(f.shape[1], sentence)
        sentence = self.context(sentence, src_key_padding_mask=~mask[rows])
        context = local.new_zeros(local.shape).index_copy(0, rows.nonzero().flatten(), sentence)
        queries = context[bidx, tidx]
        if self.config_name == 'HYBRID':
            queries = queries + cand
        q = self.query(queries)
        keys, values = self.key(context), self.value(context)
        # Per-sentence, chunked LIVE candidate queries -> full sentence token keys.
        # No T*C packed key slots, modulo mapping, candidate cap, or collisions.
        attended = torch.zeros_like(q)
        heads, d = self.cfg['heads'], self.cfg['dim']
        for row in rows.nonzero().flatten().tolist():
            live = (bidx == row).nonzero().flatten()
            n = int(lengths[row])
            k = keys[row, :n].view(n, heads, d // heads).transpose(0, 1)[None]
            v = values[row, :n].view(n, heads, d // heads).transpose(0, 1)[None]
            for chunk in live.split(self.cfg['query_chunk']):
                query = q[chunk].view(-1, heads, d // heads).transpose(0, 1)[None]
                value = F.scaled_dot_product_attention(query, k, v, dropout_p=0.0)
                attended[chunk] = value[0].transpose(0, 1).reshape(-1, d)
        interaction = self.interaction_out(attended)
        features = torch.cat((local[bidx, tidx], context[bidx, tidx], cand, interaction), -1)
        correction = self.scorer(features).squeeze(-1)
        result = base_scores.clone()
        result[bidx, tidx, cidx] = result[bidx, tidx, cidx] + correction
        return result


def load_reranker_checkpoint(path, device=None):
    """Strict full-state reload; metadata follows span_rerank_pilot conventions."""
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('arm') not in {arm_name(name) for name in CONFIGS}:
        raise ValueError('Expected hybrid_rerank_CONTROL/HYBRID checkpoint')
    if payload['arm'] != arm_name(payload['config_name']):
        raise ValueError('Checkpoint arm/config_name mismatch')
    model = HybridReranker(payload, payload['config_name'], payload['reranker_config'], load_base=False)
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device or 'cpu').eval(), payload
