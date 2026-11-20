"""Fast, zero-copy CPU/GPU restorer for ViDiCRFv2 with long-text halo & diacritic preservation."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import unicodedata

import torch

from .data_pipeline import (
    TOK,
    V2Inventory,
    case_id,
    encode_strings_chars,
    encode_strings_phono,
    extract_gaps,
)
from .model import ViDiCRFv2, trigram_marginals, trigram_viterbi
from .orthography import has_diacritic, normalize_tone_placement, strip_diac


def restore_case(candidate: str, source: str) -> str:
    """Transfer character-by-character case from source onto candidate."""
    if len(candidate) != len(source):
        return candidate
    return "".join(
        c.upper() if s.isupper() else c.lower() for c, s in zip(candidate, source)
    )


def replace_spans(source: str, matches: Sequence, words: Sequence[str]) -> str:
    if len(matches) != len(words):
        raise ValueError("Prediction/source token count mismatch")
    chunks: List[str] = []
    end = 0
    for match, word in zip(matches, words):
        chunks.extend((source[end : match.start()], word))
        end = match.end()
    return "".join(chunks) + source[end:]


class FastRestorerV2:
    """End-to-end restorer with precomputed cache, Zero-Copy OOV, and user-lock support."""

    def __init__(
        self,
        model: ViDiCRFv2,
        inventory: V2Inventory,
        device: str | torch.device = "cpu",
        use_cache: bool = True,
    ):
        self.model = model.to(device).eval()
        self.inv = inventory
        self.device = torch.device(device)
        if use_cache:
            self.model.build_inference_cache()

        # Precompute lookup map (base_id, lower_form) -> candidate slot for user-locked diacritics
        self._form_to_slot: Dict[Tuple[int, str], int] = {}
        cand_np = self.inv.cand_ids.cpu().numpy()
        for bi in range(1, len(self.inv.bases) + 1):
            for k in range(cand_np.shape[1]):
                vi = int(cand_np[bi, k])
                if vi < 0:
                    break
                self._form_to_slot[(bi, self.inv.vocab[vi])] = k

        # Build surface lexical lookup for accented input words
        self._lex_lookup: Dict[str, int] = {}
        lex_np = self.inv.cand_lex.cpu().numpy()
        for vi, lx in enumerate(lex_np):
            if lx > 0:
                self._lex_lookup[self.inv.vocab[vi]] = int(lx)

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
        """Restore diacritics on `source` without mutating any model buffers.

        Supports arbitrary-length texts via sliding context-halo windows when tokens > max_window.
        """
        matches = list(TOK.finditer(source))
        n = len(matches)
        if n == 0:
            return source, {"input_tokens": 0, "known_base_tokens": 0, "oov_copy_tokens": 0}

        _, gap_pairs = extract_gaps(source)
        gap_seq = [pre for pre, _ in gap_pairs] + [gap_pairs[-1][1]]

        raw_toks = [m.group() for m in matches]
        stripped_low = [strip_diac(t).lower() for t in raw_toks]
        base_ids = [self.inv.bid.get(b, 0) for b in stripped_low]
        case_ids = [case_id(t) for t in raw_toks]
        has_diac = [1 if has_diacritic(t) else 0 for t in raw_toks]

        # Surface word string passed to encoder: preserve existing diacritics when present
        surf_strs = []
        surf_lex = []
        for t, b, hd in zip(raw_toks, stripped_low, has_diac):
            if hd:
                s_low = unicodedata.normalize("NFC", t).lower()
                s_norm = normalize_tone_placement(s_low, style=self.inv.style)
                surf_strs.append(s_norm)
                surf_lex.append(self._lex_lookup.get(s_norm, 0))
            else:
                surf_strs.append(b)
                surf_lex.append(0)

        c2i = self.inv.char_tables["c2i"]
        all_sch = encode_strings_chars(surf_strs, c2i, self.inv.chw).to(self.device)
        all_sph = encode_strings_phono(surf_strs).to(self.device)

        locked_set = set(locked_word_indices or ())
        if preserve_input_diacritics:
            for i, hd in enumerate(has_diac):
                if hd:
                    locked_set.add(i)

        # Split into overlapping windows only if n > max_window
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
            lg_t = torch.tensor(
                [[self.inv.pid.get(g, 1) for g in gap_seq[w_start:w_end]]],
                dtype=torch.long,
                device=self.device,
            )
            rg_t = torch.tensor(
                [[self.inv.pid.get(g, 1) for g in gap_seq[w_start + 1 : w_end + 1]]],
                dtype=torch.long,
                device=self.device,
            )
            sch_t = all_sch[w_start:w_end].unsqueeze(0)
            sph_t = all_sph[w_start:w_end].unsqueeze(0)
            hd_t = torch.tensor([has_diac[w_start:w_end]], dtype=torch.long, device=self.device)
            slx_t = torch.tensor([surf_lex[w_start:w_end]], dtype=torch.long, device=self.device)

            # Determine locked slots before forward() so forced candidate embeddings
            # participate in bigram/trigram transitions even if outside Top-M emission.
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
                    s_low = unicodedata.normalize("NFC", orig).lower()
                    s_norm = normalize_tone_placement(s_low, style=self.inv.style)
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
                pred_p = trigram_viterbi(
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
        return replace_spans(source, matches, chosen_words), coverage
