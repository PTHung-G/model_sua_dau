"""Data preparation, phonological/character tables, and Context-Halo / Partial-Diacritic batching."""
from __future__ import annotations

import collections
from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple
import unicodedata

import numpy as np
import torch

from .orthography import (
    MOD_MARKS,
    TONE_MARKS,
    compact_inventory,
    has_diacritic,
    is_valid_vn_or_clean_candidate,
    normalize_tone_placement,
    parse_syllable_phonology,
    strip_diac,
)

TOK = re.compile(r"[^\W_]+", re.UNICODE)
PAD_GAP = "<PAD>"
UNK_GAP = "<UNK>"


def case_id(s: str) -> int:
    if s.islower():
        return 1
    if s.isupper():
        return 3
    if s.istitle():
        return 2
    return 4


def extract_gaps(source: str) -> Tuple[List[re.Match], List[Tuple[str, str]]]:
    ms = list(TOK.finditer(source))
    out: List[Tuple[str, str]] = []
    for i, m in enumerate(ms):
        prev = source[ms[i - 1].end() : m.start()] if i else source[: m.start()]
        nxt = source[m.end() : ms[i + 1].start()] if i + 1 < len(ms) else source[m.end() :]
        out.append(
            (
                "".join(c for c in prev if not c.isalnum()),
                "".join(c for c in nxt if not c.isalnum()),
            )
        )
    return ms, out


def build_char_tables(
    strings: Sequence[str],
    min_char_count: int = 10,
) -> dict:
    """Build character vocabulary + compositional maps (base, mod, tone) per char id."""
    counts = collections.Counter()
    for s in strings:
        counts.update(s)

    # Always include standard ASCII alphanumerics and Vietnamese characters even on small corpora
    always_keep = set("0123456789abcdefghijklmnopqrstuvwxyzđ")
    for base_v in "aeiouy":
        always_keep.add(base_v)
        for m in ("", "\u0306", "\u0302", "\u031b"):
            for t in ("", *TONE_MARKS):
                always_keep.add(unicodedata.normalize("NFC", base_v + m + t))

    chars = sorted(
        c for c, n in counts.items() if (n >= min_char_count or c in always_keep)
    )
    # 0 = PAD, 1 = UNK, 2.. = known chars
    c2i: Dict[str, int] = {c: i + 2 for i, c in enumerate(chars)}
    nchar = len(chars) + 2

    base_char_set = sorted(
        {
            (
                "đ"
                if unicodedata.normalize("NFD", c)[0].lower() == "đ"
                else unicodedata.normalize("NFD", c)[0].lower()
            )
            for c in chars
        }
    )
    bc2i = {bc: i + 2 for i, bc in enumerate(base_char_set)}
    n_base_chars = len(base_char_set) + 2

    # mod: 0=PAD, 1=none, 2=breve, 3=circumflex, 4=horn
    mod_map_char = {"\u0306": 2, "\u0302": 3, "\u031b": 4}
    # tone: 0=PAD, 1=ngang, 2=huyen, 3=sac, 4=hoi, 5=nga, 6=nang
    tone_map_char = {
        "": 1,
        "\u0300": 2,
        "\u0301": 3,
        "\u0309": 4,
        "\u0303": 5,
        "\u0323": 6,
    }

    char_base_map = np.zeros(nchar, dtype=np.int64)
    char_mod_map = np.zeros(nchar, dtype=np.int64)
    char_tone_map = np.zeros(nchar, dtype=np.int64)
    # Index 1 = UNK
    char_base_map[1] = 1
    char_mod_map[1] = 1
    char_tone_map[1] = 1

    for c, idx in c2i.items():
        nfd = unicodedata.normalize("NFD", c)
        b0 = nfd[0].lower()
        b_key = "đ" if b0 == "đ" else b0
        char_base_map[idx] = bc2i.get(b_key, 1)
        m_id = 1
        t_id = 1
        for mark in nfd[1:]:
            if mark in mod_map_char:
                m_id = mod_map_char[mark]
            elif mark in tone_map_char:
                t_id = tone_map_char[mark]
        char_mod_map[idx] = m_id
        char_tone_map[idx] = t_id

    return {
        "c2i": c2i,
        "nchar": nchar,
        "char_base_map": torch.from_numpy(char_base_map),
        "char_mod_map": torch.from_numpy(char_mod_map),
        "char_tone_map": torch.from_numpy(char_tone_map),
        "n_base_chars": n_base_chars,
        "n_char_mods": 5,
        "n_char_tones": 7,
    }


def encode_strings_chars(strings: Sequence[str], c2i: Dict[str, int], width: int) -> torch.Tensor:
    arr = np.zeros((len(strings), width), dtype=np.int16)
    for i, s in enumerate(strings):
        for j in range(min(len(s), width)):
            arr[i, j] = c2i.get(s[j], 1)
    return torch.from_numpy(arr)


def encode_strings_phono(strings: Sequence[str]) -> torch.Tensor:
    arr = np.zeros((len(strings), 6), dtype=np.uint8)
    for i, s in enumerate(strings):
        arr[i] = parse_syllable_phonology(s)
    return torch.from_numpy(arr)


@dataclass
class V2Inventory:
    bases: List[str]
    bid: Dict[str, int]
    vocab: List[str]
    vid: Dict[str, int]
    cand_ids: torch.Tensor
    cand_priors: torch.Tensor
    base_ch: torch.Tensor
    base_phono: torch.Tensor
    cand_ch: torch.Tensor
    cand_phono: torch.Tensor
    cand_lex: torch.Tensor
    cand_has_diac: torch.Tensor
    n_lex_vocab: int
    punct: List[str]
    pid: Dict[str, int]
    char_tables: dict
    chw: int
    kmax: int
    style: str
    normalize_orthography: bool
    stats: dict


def build_v2_inventory(
    train_pairs: Sequence[Tuple[str, str]],
    max_lex_vocab: int = 16384,
    min_lex_count: int = 2,
    kmax: int = 32,
    chw: int = 20,
    min_char_count: int = 10,
    max_punct: int = 8192,
    style: str = "classic",
    normalize_orthography: bool = True,
    drop_identity_singletons: bool = True,
    filter_invalid: bool = True,
    min_identity_vn_count: int = 5,
) -> V2Inventory:
    """Build compacted, orthography-normalized inventory with train priors & lexical IDs."""
    forms_by_base: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    vocab_total_counts: collections.Counter = collections.Counter()
    pc: collections.Counter = collections.Counter()

    for source, target in train_pairs:
        target_nfc = unicodedata.normalize("NFC", target)
        ms, gs = extract_gaps(source)
        gt = [m.group().lower() for m in TOK.finditer(target_nfc)]
        if len(ms) != len(gt):
            continue
        for (pre, nxt), gold_raw in zip(gs, gt):
            form = normalize_tone_placement(gold_raw, style=style) if normalize_orthography else gold_raw
            base = strip_diac(form).lower()
            if not base:
                continue
            if filter_invalid and not is_valid_vn_or_clean_candidate(form, base):
                continue
            forms_by_base[base][form] += 1
            vocab_total_counts[form] += 1
            pc[pre] += 1
            pc[nxt] += 1

    punct = [PAD_GAP, UNK_GAP, ""]
    for g, _ in pc.most_common(max(0, max_punct - 3)):
        if g not in punct:
            punct.append(g)
    pid = {g: i for i, g in enumerate(punct)}

    raw_bases = sorted(forms_by_base)
    new_bases, new_vocab, new_rows, new_counts, comp_stats = compact_inventory(
        raw_bases,
        [],
        ctab_or_counts=forms_by_base,
        style=style,
        drop_identity_singletons=drop_identity_singletons,
        filter_invalid=filter_invalid,
        kmax=kmax,
        min_identity_vn_count=min_identity_vn_count,
    )
    bid = {b: i + 1 for i, b in enumerate(new_bases)}
    vid = {v: i for i, v in enumerate(new_vocab)}

    cand_ids = torch.full((len(new_bases) + 1, kmax), -1, dtype=torch.int32)
    cand_priors = torch.zeros((len(new_bases) + 1, kmax, 3), dtype=torch.float16)

    for bi, (b, row_ids, row_cnts) in enumerate(zip(new_bases, new_rows, new_counts), 1):
        tot = sum(row_cnts)
        for k, (v_idx, cnt) in enumerate(zip(row_ids, row_cnts)):
            if v_idx < 0:
                break
            cand_ids[bi, k] = v_idx
            prob = cnt / max(1.0, tot)
            cand_priors[bi, k, 0] = math.log(prob + 1e-6)
            cand_priors[bi, k, 1] = math.log1p(cnt) / 15.0
            cand_priors[bi, k, 2] = 1.0 if new_vocab[v_idx] == b else 0.0

    # Build frequent lexical syllable vocabulary (0 reserved for rare/no-lex)
    lex_candidates = [
        w
        for w, cnt in vocab_total_counts.most_common(max_lex_vocab)
        if cnt >= min_lex_count and w in vid
    ]
    lex_map = {w: i + 1 for i, w in enumerate(lex_candidates)}
    n_lex_vocab = len(lex_map) + 1

    cand_lex = torch.zeros(len(new_vocab), dtype=torch.int32)
    cand_has_diac = torch.zeros(len(new_vocab), dtype=torch.bool)
    for i, w in enumerate(new_vocab):
        cand_lex[i] = lex_map.get(w, 0)
        cand_has_diac[i] = has_diacritic(w)

    char_tables = build_char_tables(new_bases + new_vocab, min_char_count=min_char_count)
    c2i = char_tables["c2i"]

    base_ch = encode_strings_chars([""] + new_bases, c2i, chw)
    base_phono = encode_strings_phono([""] + new_bases)
    cand_ch = encode_strings_chars(new_vocab, c2i, chw)
    cand_phono = encode_strings_phono(new_vocab)

    stats = {
        **comp_stats,
        "punct_vocab": len(punct),
        "nchar": char_tables["nchar"],
        "n_lex_vocab": n_lex_vocab,
        "kmax": kmax,
        "chw": chw,
    }
    return V2Inventory(
        bases=new_bases,
        bid=bid,
        vocab=new_vocab,
        vid=vid,
        cand_ids=cand_ids,
        cand_priors=cand_priors,
        base_ch=base_ch,
        base_phono=base_phono,
        cand_ch=cand_ch,
        cand_phono=cand_phono,
        cand_lex=cand_lex,
        cand_has_diac=cand_has_diac,
        n_lex_vocab=n_lex_vocab,
        punct=punct,
        pid=pid,
        char_tables=char_tables,
        chw=chw,
        kmax=kmax,
        style=style,
        normalize_orthography=normalize_orthography,
        stats=stats,
    )


def _split_segment_spans(
    n: int,
    gap_strs: Sequence[str],
    max_len: int = 128,
    halo: int = 16,
) -> List[Tuple[int, int, int, int]]:
    """Split a paragraph of length n into (seg_start, seg_end, core_start, core_end).

    Prefers cutting at sentence-ending punctuation ('.', '!', '?', ';') with zero overlap.
    Only falls back to sliding halo windows if a single unpunctuated run exceeds max_len.
    """
    if n <= max_len:
        return [(0, n, 0, n)]

    spans: List[Tuple[int, int, int, int]] = []
    pos = 0
    while pos < n:
        if n - pos <= max_len:
            spans.append((pos, n, pos, n))
            break
        # Look backwards from pos + max_len for terminal/clause punctuation gap
        cut = -1
        for idx in range(pos + max_len, pos + max_len // 3, -1):
            if idx < len(gap_strs) and any(p in gap_strs[idx] for p in (".", "!", "?", ";", "\n")):
                cut = idx
                break
        if cut == -1:
            for idx in range(pos + max_len, pos + max_len // 2, -1):
                if idx < len(gap_strs) and "," in gap_strs[idx]:
                    cut = idx
                    break
        if cut != -1:
            spans.append((pos, cut, pos, cut))
            pos = cut
        else:
            core_len = max(16, max_len - 2 * halo)
            c_end = min(n, pos + core_len)
            s_start = max(0, pos - halo)
            s_end = min(n, c_end + halo)
            spans.append((s_start, s_end, pos, c_end))
            pos = c_end
    return spans


@dataclass
class EncodedSplitV2:
    base_ids: np.ndarray       # (N_tok,) int32
    gold_slots: np.ndarray     # (N_tok,) int16
    gold_vids: np.ndarray      # (N_tok,) int32 (-1 if base_id == 0)
    case_ids: np.ndarray       # (N_tok,) uint8
    surf_ch: np.ndarray        # (N_tok, CHW) int16
    surf_phono: np.ndarray     # (N_tok, 6) uint8
    core_mask: np.ndarray      # (N_tok,) bool
    gaps: np.ndarray           # (N_tok + N_seg,) uint16
    seg_lens: np.ndarray       # (N_seg,) int32
    seg_tok_off: np.ndarray    # (N_seg,) int64
    seg_gap_off: np.ndarray    # (N_seg,) int64
    kept_rows: int
    skipped_rows: int


def encode_corpus_v2(
    pairs: Sequence[Tuple[str, str]],
    inv: V2Inventory,
    max_len: int = 128,
    halo: int = 16,
) -> EncodedSplitV2:
    """Encode raw (input, target) pairs into v2 tensors with punctuation-aware chunking."""
    lookup: List[Dict[int, int]] = [{} for _ in range(len(inv.bases) + 1)]
    cand_np = inv.cand_ids.numpy()
    for bi in range(1, len(inv.bases) + 1):
        for k in range(cand_np.shape[1]):
            vi = int(cand_np[bi, k])
            if vi < 0:
                break
            lookup[bi][vi] = k

    c2i = inv.char_tables["c2i"]
    all_base: List[int] = []
    all_gslot: List[int] = []
    all_gvid: List[int] = []
    all_case: List[int] = []
    all_surf_strs: List[str] = []
    all_core: List[bool] = []
    all_gaps: List[int] = []
    seg_lens: List[int] = []
    kept = skipped = 0

    for source, target in pairs:
        target_nfc = unicodedata.normalize("NFC", target)
        ms, gs = extract_gaps(source)
        gt = [m.group().lower() for m in TOK.finditer(target_nfc)]
        if not ms or len(ms) != len(gt) or len(ms) > 256:
            skipped += 1
            continue

        row_b: List[int] = []
        row_gs: List[int] = []
        row_gv: List[int] = []
        row_c: List[int] = []
        row_s: List[str] = []
        ok = True

        for m, gold_raw in zip(ms, gt):
            form = (
                normalize_tone_placement(gold_raw, style=inv.style)
                if inv.normalize_orthography
                else gold_raw
            )
            b = strip_diac(form).lower()
            bi = inv.bid.get(b, 0)
            if bi > 0:
                vi = inv.vid.get(form, -1)
                ci = lookup[bi].get(vi)
                if ci is None:
                    ok = False
                    break
                row_b.append(bi)
                row_gs.append(ci)
                row_gv.append(vi)
            else:
                # Dropped identity singleton (or OOV copy): valid iff gold == base
                if form != b:
                    ok = False
                    break
                row_b.append(0)
                row_gs.append(0)
                row_gv.append(-1)
            row_c.append(case_id(m.group()))
            row_s.append(b)

        if not ok:
            skipped += 1
            continue

        gap_strs = [pre for pre, _ in gs] + [gs[-1][1]]
        gap_ids = [inv.pid.get(g, 1) for g in gap_strs]
        spans = _split_segment_spans(len(row_b), gap_strs, max_len=max_len, halo=halo)
        for s_start, s_end, c_start, c_end in spans:
            all_base.extend(row_b[s_start:s_end])
            all_gslot.extend(row_gs[s_start:s_end])
            all_gvid.extend(row_gv[s_start:s_end])
            all_case.extend(row_c[s_start:s_end])
            all_surf_strs.extend(row_s[s_start:s_end])
            all_core.extend(c_start <= idx < c_end for idx in range(s_start, s_end))
            all_gaps.extend(gap_ids[s_start : s_end + 1])
            seg_lens.append(s_end - s_start)
        kept += 1

    seg_lens_arr = np.asarray(seg_lens, dtype=np.int32)
    seg_tok_off = np.zeros(len(seg_lens_arr), dtype=np.int64)
    seg_gap_off = np.zeros(len(seg_lens_arr), dtype=np.int64)
    if len(seg_lens_arr) > 1:
        np.cumsum(seg_lens_arr[:-1], out=seg_tok_off[1:])
        np.cumsum(seg_lens_arr[:-1] + 1, out=seg_gap_off[1:])

    # Unique encoding for surface strings (fast even on 500k+ tokens)
    uniq_s, inv_idx = np.unique(np.asarray(all_surf_strs, dtype=object), return_inverse=True) if all_surf_strs else (np.empty(0, dtype=object), np.empty(0, dtype=np.int64))
    if len(uniq_s) > 0:
        u_list = uniq_s.tolist()
        u_ch = encode_strings_chars(u_list, c2i, inv.chw).numpy()
        u_ph = encode_strings_phono(u_list).numpy()
        surf_ch = u_ch[inv_idx]
        surf_phono = u_ph[inv_idx]
    else:
        surf_ch = np.zeros((0, inv.chw), dtype=np.int16)
        surf_phono = np.zeros((0, 6), dtype=np.uint8)

    return EncodedSplitV2(
        base_ids=np.asarray(all_base, dtype=np.int32),
        gold_slots=np.asarray(all_gslot, dtype=np.int16),
        gold_vids=np.asarray(all_gvid, dtype=np.int32),
        case_ids=np.asarray(all_case, dtype=np.uint8),
        surf_ch=surf_ch,
        surf_phono=surf_phono,
        core_mask=np.asarray(all_core, dtype=bool),
        gaps=np.asarray(all_gaps, dtype=np.uint16),
        seg_lens=seg_lens_arr,
        seg_tok_off=seg_tok_off,
        seg_gap_off=seg_gap_off,
        kept_rows=kept,
        skipped_rows=skipped,
    )


def make_batch_v2(
    split: EncodedSplitV2,
    inv: V2Inventory,
    idx: np.ndarray,
    device: str | torch.device,
    partial_diac_prob: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Assemble padded batch tensors, optionally injecting stochastic partial input diacritics."""
    idx = np.asarray(idx, dtype=np.int64)
    ln = split.seg_lens[idx]
    width = int(ln.max())
    B = len(idx)

    base = np.zeros((B, width), dtype=np.int64)
    gold = np.zeros((B, width), dtype=np.int64)
    gold_vid = np.full((B, width), -1, dtype=np.int64)
    case = np.zeros((B, width), dtype=np.int64)
    left_gaps = np.zeros((B, width), dtype=np.int64)
    right_gaps = np.zeros((B, width), dtype=np.int64)
    surf_ch = np.zeros((B, width, inv.chw), dtype=np.int64)
    surf_phono = np.zeros((B, width, 6), dtype=np.int64)
    core_mask = np.zeros((B, width), dtype=bool)

    for row, seg_i in enumerate(idx):
        n = int(ln[row])
        t_off = int(split.seg_tok_off[seg_i])
        g_off = int(split.seg_gap_off[seg_i])
        base[row, :n] = split.base_ids[t_off : t_off + n]
        gold[row, :n] = split.gold_slots[t_off : t_off + n]
        gold_vid[row, :n] = split.gold_vids[t_off : t_off + n]
        case[row, :n] = split.case_ids[t_off : t_off + n]
        surf_ch[row, :n] = split.surf_ch[t_off : t_off + n]
        surf_phono[row, :n] = split.surf_phono[t_off : t_off + n]
        core_mask[row, :n] = split.core_mask[t_off : t_off + n]
        seg_gaps = split.gaps[g_off : g_off + n + 1]
        left_gaps[row, :n] = seg_gaps[:-1]
        right_gaps[row, :n] = seg_gaps[1:]

    base_t = torch.as_tensor(base, device=device)
    gold_t = torch.as_tensor(gold, device=device)
    gold_vid_t = torch.as_tensor(gold_vid, device=device)
    sent_len_t = torch.as_tensor(ln, dtype=torch.long, device=device)
    case_t = torch.as_tensor(case, device=device)
    lg_t = torch.as_tensor(left_gaps, device=device)
    rg_t = torch.as_tensor(right_gaps, device=device)
    sch_t = torch.as_tensor(surf_ch, device=device)
    sph_t = torch.as_tensor(surf_phono, device=device)
    core_t = torch.as_tensor(core_mask, device=device)
    has_diac_t = torch.zeros((B, width), dtype=torch.long, device=device)
    slex_t = torch.zeros((B, width), dtype=torch.long, device=device)

    if partial_diac_prob > 0.0 and rng is not None:
        apply_rows = rng.random(B) < partial_diac_prob
        if apply_rows.any():
            keep_np = np.zeros((B, width), dtype=bool)
            for row in np.where(apply_rows)[0]:
                p_keep = float(rng.uniform(0.20, 0.70))
                n = int(ln[row])
                keep_np[row, :n] = rng.random(n) < p_keep
            keep_t = torch.as_tensor(keep_np, device=device) & (gold_vid_t >= 0)
            valid_gvid = gold_vid_t.clamp(min=0)
            # Only set has_input_diac where the gold candidate actually carries diacritics
            cand_has = inv.cand_has_diac.to(device)[valid_gvid]
            keep_diac = keep_t & cand_has
            if keep_diac.any():
                c_ch = inv.cand_ch.to(device).long()[valid_gvid]
                c_ph = inv.cand_phono.to(device).long()[valid_gvid]
                c_lx = inv.cand_lex.to(device).long()[valid_gvid]
                sch_t = torch.where(keep_diac.unsqueeze(-1), c_ch, sch_t)
                sph_t = torch.where(keep_diac.unsqueeze(-1), c_ph, sph_t)
                slex_t = torch.where(keep_diac, c_lx, slex_t)
                has_diac_t = keep_diac.long()

    return {
        "base_ids": base_t,
        "gold": gold_t,
        "sent_len": sent_len_t,
        "case_ids": case_t,
        "left_gaps": lg_t,
        "right_gaps": rg_t,
        "surface_chars": sch_t,
        "surface_phono": sph_t,
        "surface_lex": slex_t,
        "has_input_diac": has_diac_t,
        "core_mask": core_t,
    }
