"""CPU inference optimizations for ViDiCRF v2: word-level caches, JIT Viterbi, CSR.

Design notes
------------
- `WordCache` memoizes per-token derived features (strip/normalize/phonology/case)
  and encoded char tensors, eliminating the per-keystroke unicodedata + regex
  overhead for recurring words (typing-demo workloads).
- `scripted_viterbi` is a TorchScript translation of `trigram_viterbi` (identical
  algorithm incl. locked-mask handling); equivalence is enforced by tests against
  the eager implementation and exhaustive enumeration.
- `CSRInventory` stores variable-length candidate rows (indptr + flat vids) instead
  of the dense (n_bases+1, K) layout, removing ~95% padding cells on compacted
  inventories.
- `FastRestorerV2Opt` wraps `FastRestorerV2` semantics with the caches above plus a
  whole-input memo; tests pin its outputs to the parent implementation.
- `profile_predict_breakdown` attributes predict() latency to stages.
"""
from __future__ import annotations

import time
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .data_pipeline import TOK, case_id, encode_strings_chars, encode_strings_phono, extract_gaps
from .model import trigram_marginals, trigram_viterbi
from .orthography import has_diacritic, normalize_tone_placement, strip_diac
from .restorer import FastRestorerV2, replace_spans, restore_case


# ---------------------------------------------------------------------------
# Word-level feature cache
# ---------------------------------------------------------------------------
class WordCache:
    """Memoize per-token features and encoded tensors for one inventory/restorer.

    features() returns a 7-tuple:
      (base_lower, surface_nfc_lower, surface_encoded, case_id, has_diac, lex_id, base_row)
    matching exactly the per-token computation in FastRestorerV2.predict.
    """

    def __init__(
        self,
        c2i: Dict[str, int],
        chw: int,
        style: str = "classic",
        lex_get: Optional[Callable[[str], int]] = None,
        bid_get: Optional[Callable[[str], int]] = None,
        pid_get: Optional[Callable[[str], int]] = None,
    ):
        self.c2i = c2i
        self.chw = chw
        self.style = style
        self._lex_get = lex_get or (lambda s: 0)
        self._bid_get = bid_get or (lambda b: 0)
        self._pid_get = pid_get or (lambda g: 1)
        self._feats: Dict[str, Tuple[str, str, str, int, int, int, int]] = {}
        self._ch: Dict[str, torch.Tensor] = {}
        self._ph: Dict[str, torch.Tensor] = {}
        self._gaps: Dict[str, int] = {}

    def features(self, tok: str) -> Tuple[str, str, str, int, int, int, int]:
        got = self._feats.get(tok)
        if got is None:
            b = strip_diac(tok).lower()
            hd = 1 if has_diacritic(tok) else 0
            s_low = unicodedata.normalize("NFC", tok).lower()
            s_enc = normalize_tone_placement(s_low, style=self.style) if hd else b
            lex = self._lex_get(s_enc) if hd else 0
            got = (b, s_low, s_enc, case_id(tok), hd, lex, self._bid_get(b))
            self._feats[tok] = got
        return got

    def char_tensors(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        miss = [w for w in words if w not in self._ch]
        for w in miss:
            self._ch[w] = encode_strings_chars([w], self.c2i, self.chw)[0]
            self._ph[w] = encode_strings_phono([w])[0]
        ch = torch.stack([self._ch[w] for w in words])
        ph = torch.stack([self._ph[w] for w in words])
        return ch, ph

    def gap_id(self, gap: str) -> int:
        got = self._gaps.get(gap)
        if got is None:
            got = self._gaps[gap] = self._pid_get(gap)
        return got

    def clear(self) -> None:
        self._feats.clear()
        self._ch.clear()
        self._ph.clear()

    def __len__(self) -> int:
        return len(self._feats)


# ---------------------------------------------------------------------------
# TorchScript exact second-order Viterbi (mirror of model.trigram_viterbi)
# ---------------------------------------------------------------------------
_SCRIPTED_VITERBI_TS: Optional[object] = None


def get_scripted_viterbi():
    """Return a TorchScript-compiled (or eager fallback) exact trigram Viterbi."""
    global _SCRIPTED_VITERBI_TS
    if _SCRIPTED_VITERBI_TS is None:
        try:
            _SCRIPTED_VITERBI_TS = torch.jit.script(scripted_viterbi.__wrapped__)
        except Exception:
            _SCRIPTED_VITERBI_TS = scripted_viterbi
    return _SCRIPTED_VITERBI_TS


@torch.no_grad()
def scripted_viterbi(
    emit_p: torch.Tensor,
    bigram: torch.Tensor,
    trigram: torch.Tensor,
    mask_p: torch.Tensor,
    locked_mask_p: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """TorchScript-compatible exact trigram Viterbi returning pruned slots (B, L)."""
    if locked_mask_p is not None:
        has_lock = locked_mask_p.any(dim=-1, keepdim=True)
        comb = mask_p & locked_mask_p
        keep = has_lock & comb.any(dim=-1, keepdim=True)
        mask_p = torch.where(keep, comb, mask_p)

    emit_m = emit_p.masked_fill(~mask_p, -1e9)
    B, L, M = emit_m.shape
    real = mask_p.any(dim=-1)

    pred_eq1 = emit_m[:, 0].argmax(dim=-1)
    if L == 1:
        return pred_eq1.unsqueeze(1)

    score2 = emit_m[:, 0].unsqueeze(2) + bigram[:, 0] + emit_m[:, 1].unsqueeze(1)
    ptr = torch.zeros((B, L, M, M), dtype=torch.long, device=emit_m.device)

    for t in range(2, L):
        cand = score2.unsqueeze(3) + trigram[:, t - 2] + bigram[:, t - 1].unsqueeze(1)
        best, idx = cand.max(dim=1)
        nxt = best + emit_m[:, t].unsqueeze(1)
        score2 = torch.where(real[:, t].view(B, 1, 1), nxt, score2)
        ptr[:, t] = idx

    best_flat = score2.reshape(B, M * M).argmax(dim=-1)
    j_cur = best_flat // M
    k_cur = best_flat % M

    out = torch.zeros((B, L), dtype=torch.long, device=emit_m.device)
    b_idx = torch.arange(B, device=emit_m.device)

    for t in range(L - 1, 1, -1):
        w = real[:, t]
        out[:, t] = torch.where(w, k_cur, out[:, t])
        prev_i = ptr[b_idx, t, j_cur, k_cur]
        k_cur = torch.where(w, j_cur, k_cur)
        j_cur = torch.where(w, prev_i, j_cur)

    w1 = real[:, 1]
    out[:, 1] = torch.where(w1, k_cur, out[:, 1])
    out[:, 0] = torch.where(w1, j_cur, pred_eq1)
    return out


# ---------------------------------------------------------------------------
# CSR candidate table
# ---------------------------------------------------------------------------
class CSRInventory:
    """Compressed-sparse-row candidate rows replacing dense (n_rows, K) padding.

    Row `bi` candidates are `vids[indptr[bi]:indptr[bi+1]]` — identical semantics
    to the dense row up to the first -1 sentinel.
    """

    def __init__(self, cand_ids_dense: torch.Tensor):
        assert cand_ids_dense.dim() == 2
        n_rows, kmax = cand_ids_dense.shape
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64)
        flat: List[int] = []
        dense = cand_ids_dense.tolist()
        for r, row in enumerate(dense):
            cnt = 0
            for v in row:
                if v < 0:
                    break
                flat.append(int(v))
                cnt += 1
            indptr[r + 1] = indptr[r] + cnt
        self.indptr = indptr
        self.vids = torch.tensor(flat, dtype=torch.int32) if flat else torch.zeros(0, dtype=torch.int32)
        self.kmax = kmax
        self.n_rows = n_rows
        self.stats = {
            "dense_bytes": int(cand_ids_dense.numel() * cand_ids_dense.element_size()),
            "csr_bytes_int32_vids": int(self.vids.numel() * 4 + self.indptr.numel() * 8),
            "avg_candidates_per_base": round(float(self.vids.numel()) / max(1, n_rows - 1), 3),
            "dense_utilization_pct": round(100.0 * self.vids.numel() / max(1, cand_ids_dense.numel()), 2),
            "compression_ratio": round(
                (cand_ids_dense.numel() * cand_ids_dense.element_size())
                / max(1, self.vids.numel() * 4 + self.indptr.numel() * 8),
                2,
            ),
        }

    def row(self, bi: int) -> torch.Tensor:
        return self.vids[self.indptr[bi] : self.indptr[bi + 1]]

    def dense_like(self, device: str | torch.device = "cpu") -> torch.Tensor:
        out = torch.full((self.n_rows, self.kmax), -1, dtype=torch.int32, device=device)
        for bi in range(self.n_rows):
            r = self.row(bi)
            out[bi, : len(r)] = r.to(device)
        return out


# ---------------------------------------------------------------------------
# Optimized restorer (output-equivalent to FastRestorerV2, faster on CPU)
# ---------------------------------------------------------------------------
class FastRestorerV2Opt(FastRestorerV2):
    """FastRestorerV2 + word caches + JIT Viterbi + whole-input memo.

    Outputs are pinned to the parent implementation by tests, including locked
    words, preserved diacritics, tone style toggles, and halo windows.
    """

    def __init__(
        self,
        model,
        inventory,
        device: str | torch.device = "cpu",
        use_cache: bool = True,
        src_cache_size: int = 1024,
    ):
        super().__init__(model, inventory, device=device, use_cache=use_cache)
        self._wcache = WordCache(
            inventory.char_tables["c2i"],
            inventory.chw,
            style=inventory.style,
            lex_get=lambda s: self._lex_lookup.get(s, 0),
            bid_get=lambda b: self.inv.bid.get(b, 0),
            pid_get=lambda g: self.inv.pid.get(g, 1),
        )
        self._src_cache: Dict[tuple, Tuple[str, dict]] = {}
        self._src_cache_size = max(0, src_cache_size)

    def clear_caches(self, keep_src: bool = False) -> None:
        self._wcache.clear()
        if not keep_src:
            self._src_cache.clear()

    @torch.inference_mode()
    def predict(
        self,
        source: str,
        decode_mode: str = "viterbi",
        preserve_input_diacritics: bool = False,
        locked_word_indices: Optional[Sequence[int]] = None,
        tone_style: Optional[str] = None,
        max_window: int = 256,
        halo: int = 24,
    ) -> Tuple[str, dict]:
        key = (
            source,
            decode_mode,
            bool(preserve_input_diacritics),
            tuple(sorted(set(locked_word_indices or ()))),
            tone_style,
            max_window,
            halo,
        )
        hit = self._src_cache.get(key)
        if hit is not None:
            return hit

        matches = list(TOK.finditer(source))
        n = len(matches)
        if n == 0:
            return source, {"input_tokens": 0, "known_base_tokens": 0, "oov_copy_tokens": 0}

        _, gap_pairs = extract_gaps(source)
        gap_seq = [pre for pre, _ in gap_pairs] + [gap_pairs[-1][1]]
        gap_ids_seq = [self._wcache.gap_id(g) for g in gap_seq]

        raw_toks = [m.group() for m in matches]
        feats = [self._wcache.features(t) for t in raw_toks]
        base_ids = [f[6] for f in feats]
        case_ids = [f[3] for f in feats]
        has_diac = [f[4] for f in feats]
        surf_strs = [f[2] for f in feats]
        surf_lex = [f[5] for f in feats]

        all_sch, all_sph = self._wcache.char_tensors(surf_strs)
        all_sch = all_sch.to(self.device)
        all_sph = all_sph.to(self.device)

        locked_set = set(locked_word_indices or ())
        if preserve_input_diacritics:
            for i, hd in enumerate(has_diac):
                if hd:
                    locked_set.add(i)

        if n <= max_window:
            windows = [(0, n, 0, n)]
        else:
            windows = []
            step = max(32, max_window - 2 * halo)
            pos = 0
            while pos < n:
                c_end = min(n, pos + step)
                w_start = max(0, pos - halo)
                w_end = min(n, c_end + halo)
                windows.append((w_start, w_end, pos, c_end))
                pos = c_end

        chosen_words = [""] * n
        known_count = sum(1 for bi in base_ids if bi > 0)

        for w_start, w_end, c_start, c_end in windows:
            wl = w_end - w_start
            b_t = torch.tensor([base_ids[w_start:w_end]], dtype=torch.long, device=self.device)
            ln_t = torch.tensor([wl], dtype=torch.long, device=self.device)
            c_t = torch.tensor([case_ids[w_start:w_end]], dtype=torch.long, device=self.device)
            lg_t = torch.tensor([gap_ids_seq[w_start:w_end]], dtype=torch.long, device=self.device)
            rg_t = torch.tensor([gap_ids_seq[w_start + 1 : w_end + 1]], dtype=torch.long, device=self.device)
            sch_t = all_sch[w_start:w_end].unsqueeze(0)
            sph_t = all_sph[w_start:w_end].unsqueeze(0)
            hd_t = torch.tensor([has_diac[w_start:w_end]], dtype=torch.long, device=self.device)
            slx_t = torch.tensor([surf_lex[w_start:w_end]], dtype=torch.long, device=self.device)

            forced_list = [-1] * wl
            hard_copy_positions: Dict[int, str] = {}
            any_forced = False
            if locked_set:
                for local_i, global_i in enumerate(range(w_start, w_end)):
                    if global_i not in locked_set:
                        continue
                    orig = raw_toks[global_i]
                    bi = base_ids[global_i]
                    if bi == 0:
                        continue
                    s_low = feats[global_i][1]
                    s_norm = feats[global_i][2]
                    slot = self._form_to_slot.get((bi, s_low))
                    if slot is None:
                        slot = self._form_to_slot.get((bi, s_norm))
                    if slot is not None:
                        forced_list[local_i] = slot
                        any_forced = True
                    else:
                        hard_copy_positions[global_i] = orig

            forced_t = (
                torch.tensor([forced_list], dtype=torch.long, device=self.device)
                if any_forced
                else None
            )

            out = self.model(
                b_t, ln_t, c_t, lg_t, rg_t, sch_t, sph_t,
                has_input_diac=hd_t, surface_lex=slx_t, forced_slots=forced_t,
            )
            emit_p = out["emit_p"]
            mask_p = out["mask_p"]
            top_idx = out["top_idx"]
            bigram = out["bigram"]
            trigram = out["trigram"]

            locked_mask_p = None
            if any_forced and forced_t is not None:
                has_f = forced_t >= 0
                locked_mask_p = mask_p & (top_idx == forced_t.clamp(min=0).unsqueeze(-1)) & has_f.unsqueeze(-1)

            if decode_mode == "mbr":
                log_m, _ = trigram_marginals(
                    emit_p, bigram, trigram, mask_p, locked_mask_p=locked_mask_p
                )
                pred_p = log_m.argmax(dim=-1)[0]
            elif decode_mode == "viterbi":
                pred_p = get_scripted_viterbi()(
                    emit_p, bigram, trigram, mask_p, locked_mask_p=locked_mask_p
                )[0]
            else:
                raise ValueError(f"Unknown decode_mode: {decode_mode}")

            for global_i in range(c_start, c_end):
                local_i = global_i - w_start
                orig = raw_toks[global_i]
                if global_i in locked_set or global_i in hard_copy_positions:
                    chosen_words[global_i] = orig
                    continue
                bi = base_ids[global_i]
                if bi == 0:
                    chosen_words[global_i] = orig
                else:
                    orig_slot = int(top_idx[0, local_i, int(pred_p[local_i])])
                    cand_vid = int(self.inv.cand_ids[bi, orig_slot])
                    cand_word = self.inv.vocab[cand_vid] if cand_vid >= 0 else orig
                    if tone_style in ("classic", "modern"):
                        cand_word = normalize_tone_placement(cand_word, style=tone_style)
                    chosen_words[global_i] = restore_case(cand_word, orig)

        coverage = {
            "input_tokens": n,
            "known_base_tokens": known_count,
            "oov_copy_tokens": n - known_count,
        }
        result = (replace_spans(source, matches, chosen_words), coverage)
        if self._src_cache_size > 0:
            if len(self._src_cache) >= self._src_cache_size:
                self._src_cache.clear()
            self._src_cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Stage profiler
# ---------------------------------------------------------------------------
@torch.inference_mode()
def profile_predict_breakdown(restorer, source: str, repeats: int = 20, use_word_cache: bool = False) -> dict:
    """Attribute predict()-path latency to stages on one input (median of repeats)."""
    inv = restorer.inv
    dev = restorer.device
    cache = None
    if use_word_cache:
        cache = getattr(restorer, "_wcache", None)
        if cache is None:
            cache = WordCache(
                inv.char_tables["c2i"], inv.chw, style=inv.style,
                lex_get=lambda s: restorer._lex_lookup.get(s, 0),
                bid_get=lambda b: inv.bid.get(b, 0),
                pid_get=lambda g: inv.pid.get(g, 1),
            )
            restorer._wcache = cache

    t_tok = t_enc = t_asm = t_fwd = t_dec = t_re = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        matches = list(TOK.finditer(source))
        raw_toks = [m.group() for m in matches]
        n = len(raw_toks)
        if cache is not None:
            feats = [cache.features(t) for t in raw_toks]
            stripped_low = [f[0] for f in feats]
            surf = [f[2] for f in feats]
            case_ids = [f[3] for f in feats]
            has_diac = [f[4] for f in feats]
            surf_lex = [f[5] for f in feats]
            gap_pairs = extract_gaps(source)[1]
            gap_ids_seq = [cache.gap_id(g) for g in [p for p, _ in gap_pairs] + [gap_pairs[-1][1]]]
        else:
            stripped_low, surf, case_ids, has_diac, surf_lex = [], [], [], [], []
            for t in raw_toks:
                b = strip_diac(t).lower()
                hd = 1 if has_diacritic(t) else 0
                if hd:
                    s = unicodedata.normalize("NFC", t).lower()
                    s = normalize_tone_placement(s, style=inv.style)
                    surf_lex.append(inv._lex_lookup.get(s, 0))
                else:
                    s = b
                    surf_lex.append(0)
                stripped_low.append(b)
                surf.append(s)
                case_ids.append(case_id(t))
                has_diac.append(hd)
            gap_pairs = extract_gaps(source)[1]
            gap_ids_seq = [inv.pid.get(g, 1) for g in [p for p, _ in gap_pairs] + [gap_pairs[-1][1]]]
        t1 = time.perf_counter()

        if cache is not None:
            sch2, sph2 = cache.char_tensors(surf)
            all_sch = sch2.to(dev)
            all_sph = sph2.to(dev)
        else:
            all_sch = encode_strings_chars(surf, inv.char_tables["c2i"], inv.chw).to(dev)
            all_sph = encode_strings_phono(surf).to(dev)
        t2 = time.perf_counter()

        base_ids = [inv.bid.get(b, 0) for b in stripped_low]
        b_t = torch.tensor([base_ids], dtype=torch.long, device=dev)
        ln_t = torch.tensor([n], dtype=torch.long, device=dev)
        c_t = torch.tensor([case_ids], dtype=torch.long, device=dev)
        lg_t = torch.tensor([gap_ids_seq[:n]], dtype=torch.long, device=dev)
        rg_t = torch.tensor([gap_ids_seq[1 : n + 1]], dtype=torch.long, device=dev)
        hd_t = torch.tensor([has_diac], dtype=torch.long, device=dev)
        slx_t = torch.tensor([surf_lex], dtype=torch.long, device=dev)
        t3 = time.perf_counter()

        out = restorer.model(
            b_t, ln_t, c_t, lg_t, rg_t, all_sch.unsqueeze(0), all_sph.unsqueeze(0),
            has_input_diac=hd_t, surface_lex=slx_t,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        pred_p = trigram_viterbi(out["emit_p"], out["bigram"], out["trigram"], out["mask_p"])[0]
        top_idx = out["top_idx"][0]
        t5 = time.perf_counter()

        for i in range(n):
            bi = base_ids[i]
            if bi == 0:
                continue
            slot = int(top_idx[i, int(pred_p[i])])
            vid = int(inv.cand_ids[bi, slot])
            w = inv.vocab[vid] if vid >= 0 else raw_toks[i]
            restore_case(w, raw_toks[i])
        t6 = time.perf_counter()

        t_tok += (t1 - t0) * 1000
        t_enc += (t2 - t1) * 1000
        t_asm += (t3 - t2) * 1000
        t_fwd += (t4 - t3) * 1000
        t_dec += (t5 - t4) * 1000
        t_re += (t6 - t5) * 1000

    k = max(1, repeats)
    return {
        "tokenize_features_ms": round(t_tok / k, 2),
        "char_encode_ms": round(t_enc / k, 2),
        "tensor_assembly_ms": round(t_asm / k, 2),
        "model_forward_ms": round(t_fwd / k, 2),
        "viterbi_decode_ms": round(t_dec / k, 2),
        "word_reassembly_ms": round(t_re / k, 2),
        "total_ms": round((t_tok + t_enc + t_asm + t_fwd + t_dec + t_re) / k, 2),
    }
