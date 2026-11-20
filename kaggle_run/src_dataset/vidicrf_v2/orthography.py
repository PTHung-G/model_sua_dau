"""Vietnamese orthography normalization, phonological decomposition, and lossless inventory compaction."""
from __future__ import annotations

import collections
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

TONE_MARKS: Dict[str, str] = {
    "\u0300": "huyen",
    "\u0301": "sac",
    "\u0309": "hoi",
    "\u0303": "nga",
    "\u0323": "nang",
}
TONE_TO_ID: Dict[str, int] = {
    "": 0,
    "huyen": 1,
    "sac": 2,
    "hoi": 3,
    "nga": 4,
    "nang": 5,
    "<OTHER>": 6,
}

MOD_MARKS: Dict[str, str] = {
    "\u0306": "breve",
    "\u0302": "circumflex",
    "\u031b": "horn",
}

VN_BASE_LETTERS = set("abcdeghiklmnopqrstuvxy")
VN_VOWELS = set("aeiouy")

# Fixed phonological vocabularies (0 reserved for PAD).
ONSETS = [
    "<PAD>", "<OTHER>", "",
    "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh",
    "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s",
    "t", "th", "tr", "v", "x",
]
GLIDES = ["<PAD>", "<OTHER>", "", "o", "u"]
NUCLEI = ["<PAD>", "<OTHER>", "", "a", "e", "i", "o", "u", "y", "ia", "ie", "ua", "uo", "ye"]
CODAS = ["<PAD>", "<OTHER>", "", "c", "ch", "i", "m", "n", "ng", "nh", "o", "p", "t", "u", "y"]
MOD_PATTERNS = ["<PAD>", "<OTHER>", "", "breve", "circumflex", "horn", "double_horn"]

ONSET_TO_ID = {k: i for i, k in enumerate(ONSETS)}
GLIDE_TO_ID = {k: i for i, k in enumerate(GLIDES)}
NUCLEUS_TO_ID = {k: i for i, k in enumerate(NUCLEI)}
CODA_TO_ID = {k: i for i, k in enumerate(CODAS)}
MOD_TO_ID = {k: i for i, k in enumerate(MOD_PATTERNS)}

PHONO_CARDINALITIES = (
    len(ONSETS),
    len(GLIDES),
    len(NUCLEI),
    len(CODAS),
    len(MOD_PATTERNS),
    len(TONE_TO_ID),
)


def strip_diac(text: str) -> str:
    """Remove combining marks and map đ/Đ -> d/D, preserving 1:1 NFC character count for VN."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.replace("đ", "d").replace("Đ", "D")


def has_diacritic(word: str) -> bool:
    """Return True if word contains any Vietnamese vowel modifier, tone mark, or đ/Đ."""
    if "đ" in word or "Đ" in word:
        return True
    for ch in unicodedata.normalize("NFD", word):
        if ch in TONE_MARKS or ch in MOD_MARKS:
            return True
    return False


def _decompose_chars(word: str) -> Tuple[bool, List[Tuple[str, str, str, str]]]:
    """Decompose NFC word into per-character (orig_base_char, lower_plain, mod_char, tone_char).

    Returns (is_pure_vn_script, slots). is_pure_vn_script is False if any character uses
    non-Vietnamese letters (e.g. f/j/w/z, digits, Greek/CJK) or non-Vietnamese marks.
    """
    nfc = unicodedata.normalize("NFC", word)
    if not nfc:
        return False, []
    slots: List[Tuple[str, str, str, str]] = []
    pure = True
    for ch in nfc:
        nfd = unicodedata.normalize("NFD", ch)
        base_char = nfd[0]
        lower_b = base_char.lower()
        lower_plain = "d" if lower_b == "đ" else lower_b
        mods = []
        tones = []
        for mark in nfd[1:]:
            if mark in MOD_MARKS:
                mods.append(mark)
            elif mark in TONE_MARKS:
                tones.append(mark)
            elif unicodedata.combining(mark):
                pure = False
        if lower_b != "đ" and lower_plain not in VN_BASE_LETTERS:
            pure = False
        if len(mods) > 1 or len(tones) > 1:
            pure = False
        slots.append((
            base_char,
            lower_plain,
            mods[0] if mods else "",
            tones[0] if tones else "",
        ))
    return pure, slots


def _find_vowel_span(plain: str, mods: Sequence[str]) -> Tuple[int, int]:
    """Locate (v_start, v_end) of the syllable vowel cluster, respecting qu- and gi- onsets."""
    n = len(plain)
    start = 0
    if n >= 3 and plain.startswith("qu") and plain[2] in VN_VOWELS and mods[1] == "":
        start = 2
    elif n >= 3 and plain.startswith("gi") and plain[2] in VN_VOWELS and mods[1] == "":
        start = 2
    v_start = start
    while v_start < n and plain[v_start] not in VN_VOWELS:
        v_start += 1
    v_end = v_start
    while v_end < n and plain[v_end] in VN_VOWELS:
        v_end += 1
    return v_start, v_end


def normalize_tone_placement(word: str, style: str = "classic") -> str:
    """Normalize Vietnamese tone placement without changing stripped base or character count.

    Fixes misplaced tones on `qu-` (qúy -> quý), `gi-` (gìa -> già), closed/circumflex
    syllables (vịêt -> việt, tòan -> toàn), and standardizes open `oa/oe/uy` diphthongs
    between `classic` (hòa, khỏe, thủy) and `modern` (hoà, khoẻ, thuỷ).
    Foreign/mixed/multi-tone words are returned unchanged (NFC).
    """
    if style not in ("classic", "modern"):
        raise ValueError(f"Unsupported style: {style}")
    nfc = unicodedata.normalize("NFC", word)
    pure, slots = _decompose_chars(nfc)
    if not pure or not slots:
        return nfc

    tone_indices = [i for i, (_, _, _, t) in enumerate(slots) if t]
    if len(tone_indices) != 1:
        return nfc
    tone_char = slots[tone_indices[0]][3]
    plain = "".join(s[1] for s in slots)
    mods = [s[2] for s in slots]

    v_start, v_end = _find_vowel_span(plain, mods)
    if v_start >= v_end:
        return nfc

    # Vietnamese syllables have a single contiguous vowel cluster; multi-syllable
    # foreign words like "Napoléon" contain later vowels after consonants -> leave intact.
    if any(plain[j] in VN_VOWELS for j in range(v_end, len(plain))):
        return nfc

    # Validate no stray modifiers outside the vowel span.
    for i in range(len(slots)):
        if (i < v_start or i >= v_end) and mods[i]:
            return nfc

    v_len = v_end - v_start
    has_coda = v_end < len(plain)
    mod_positions = [i for i in range(v_start, v_end) if mods[i]]

    if mod_positions:
        # Last modified vowel handles ă, â, ê, ô, ơ, ư and ươ (tone on ơ), iê (ê), uô (ô).
        target_idx = mod_positions[-1]
    elif v_len == 1:
        target_idx = v_start
    elif v_len >= 3:
        # Unmodified triphthongs (oai, oay, uyu, uya...): tone on middle vowel.
        target_idx = v_start + 1
    else:
        # 2 unmodified vowels
        pair = plain[v_start:v_end]
        if has_coda:
            target_idx = v_start + 1
        elif pair in ("oa", "oe", "uy"):
            target_idx = v_start if style == "classic" else (v_start + 1)
        elif pair in ("ai", "ay", "ao", "au", "eo", "ia", "iu", "oi", "ua", "ui"):
            target_idx = v_start
        else:
            # Foreign/pinyin pairs like "guó", "duo" -> leave untouched
            return nfc

    out_chars = []
    for i, (base_char, _, mod_char, _) in enumerate(slots):
        t = tone_char if i == target_idx else ""
        out_chars.append(unicodedata.normalize("NFC", base_char + mod_char + t))
    result = "".join(out_chars)
    if len(result) != len(nfc) or strip_diac(result) != strip_diac(nfc):
        return nfc
    return result


def is_valid_vn_or_clean_candidate(form: str, base: str | None = None) -> bool:
    """Filter obvious orthographic corruption (multi-tone syllables, invalid length/Turkish i-dot)."""
    nfc = unicodedata.normalize("NFC", form)
    if not nfc:
        return False
    if base is not None:
        base_nfc = unicodedata.normalize("NFC", base)
        if strip_diac(nfc).lower() != base_nfc.lower():
            return False
        if len(nfc) != len(base_nfc):
            return False
    pure, slots = _decompose_chars(nfc)
    tones = [s[3] for s in slots if s[3]]
    if len(tones) > 1:
        return False
    if pure and tones:
        plain = "".join(s[1] for s in slots)
        mods = [s[2] for s in slots]
        v_start, v_end = _find_vowel_span(plain, mods)
        if v_start >= v_end:
            return False
        # Tone must sit on a vowel in the vowel nucleus span.
        tone_pos = next(i for i, s in enumerate(slots) if s[3])
        if not (v_start <= tone_pos < v_end):
            return False
    return True


def parse_syllable_phonology(word: str) -> Tuple[int, int, int, int, int, int]:
    """Decompose a syllable into 6 integer IDs: (onset, glide, nucleus, coda, mod_pattern, tone)."""
    nfc = unicodedata.normalize("NFC", word)
    pure, slots = _decompose_chars(nfc)
    if not pure or not slots:
        # Check if any single VN tone exists on a mixed token; otherwise <OTHER>.
        tones = [TONE_MARKS[c] for c in unicodedata.normalize("NFD", nfc) if c in TONE_MARKS]
        tone_id = TONE_TO_ID[tones[0]] if len(tones) == 1 else TONE_TO_ID["<OTHER>"]
        return (
            ONSET_TO_ID["<OTHER>"],
            GLIDE_TO_ID["<OTHER>"],
            NUCLEUS_TO_ID["<OTHER>"],
            CODA_TO_ID["<OTHER>"],
            MOD_TO_ID["<OTHER>"],
            tone_id,
        )

    tones = [s[3] for s in slots if s[3]]
    if len(tones) == 0:
        tone_id = TONE_TO_ID[""]
    elif len(tones) == 1:
        tone_id = TONE_TO_ID[TONE_MARKS[tones[0]]]
    else:
        tone_id = TONE_TO_ID["<OTHER>"]

    plain = "".join("đ" if s[0].lower() == "đ" else s[1] for s in slots)
    plain_ascii = "".join(s[1] for s in slots)
    mods = [s[2] for s in slots]

    v_start, v_end = _find_vowel_span(plain_ascii, mods)
    onset = plain[:v_start]
    coda = plain[v_end:]
    vowel_str = plain_ascii[v_start:v_end]
    vowel_mods = [MOD_MARKS[m] for m in mods[v_start:v_end] if m]

    # Split optional medial glide (o/u) in clusters like oa, oe, uy, oai, uye, uan...
    glide = ""
    nucleus = vowel_str
    if len(vowel_str) >= 2 and onset != "qu":
        if vowel_str[0] == "o" and vowel_str[1] in ("a", "e") and mods[v_start] == "":
            glide = "o"
            nucleus = vowel_str[1:]
        elif vowel_str[0] == "u" and vowel_str[1] in ("y", "e", "a", "o") and mods[v_start] == "":
            # Distinguish uô (nucleus "uo") and ua (open "ua" without coda) from medial glide u (uê, uâ, uy, uya...)
            if not (mods[v_start + 1] == "\u0302" and vowel_str[1] == "o") and not (vowel_str == "ua" and not coda):
                glide = "u"
                nucleus = vowel_str[1:]

    # Separate trailing semi-vowel coda (i, y, u, o) from nucleus if length >= 2
    if not coda and len(nucleus) >= 2 and nucleus not in ("ia", "ua") and nucleus[-1] in ("i", "y", "u", "o"):
        # e.g., ai -> a + i, ieu -> ie + u, uoi -> uo + i, oai -> glide=o, nucleus=a, coda=i
        if not (nucleus in ("ie", "uo", "ye")):
            coda = nucleus[-1]
            nucleus = nucleus[:-1]

    if not vowel_mods:
        mod_key = ""
    elif vowel_mods == ["horn", "horn"]:
        mod_key = "double_horn"
    elif len(vowel_mods) == 1:
        mod_key = vowel_mods[0]
    else:
        mod_key = "<OTHER>"

    return (
        ONSET_TO_ID.get(onset, ONSET_TO_ID["<OTHER>"]),
        GLIDE_TO_ID.get(glide, GLIDE_TO_ID["<OTHER>"]),
        NUCLEUS_TO_ID.get(nucleus, NUCLEUS_TO_ID["<OTHER>"]),
        CODA_TO_ID.get(coda, CODA_TO_ID["<OTHER>"]),
        MOD_TO_ID.get(mod_key, MOD_TO_ID["<OTHER>"]),
        tone_id,
    )


def compact_inventory(
    bases: Sequence[str],
    vocab: Sequence[str],
    ctab_or_counts=None,
    style: str = "classic",
    drop_identity_singletons: bool = True,
    filter_invalid: bool = True,
    kmax: int = 32,
    min_identity_vn_count: int = 0,
) -> Tuple[List[str], List[str], List[List[int]], List[List[float]], dict]:
    """Normalize tone placement, filter invalid corruptions, and drop redundant identity singletons.

    Returns (new_bases, new_vocab, new_cand_rows, new_prior_probs, stats).
    `ctab_or_counts` can be either a 2D int array/list `ctab` (with base 0 padding, bases 1..N)
    or a dict `base -> Counter(form -> count)`.
    """
    form_counts_by_base: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    if isinstance(ctab_or_counts, dict):
        iterator = ctab_or_counts.items()
        for base, cnt in iterator:
            b_low = base.lower()
            for form, c in cnt.items():
                if filter_invalid and not is_valid_vn_or_clean_candidate(form, b_low):
                    continue
                norm_form = normalize_tone_placement(form.lower(), style=style)
                form_counts_by_base[b_low][norm_form] += int(c)
    else:
        # Reconstruct from (bases, vocab, ctab) preserving rank order when counts unavailable.
        for bi, base in enumerate(bases, 1):
            b_low = base.lower()
            if ctab_or_counts is not None:
                row = ctab_or_counts[bi]
                cands = [vocab[int(vi)] for vi in row if int(vi) >= 0]
            else:
                cands = []
            for rank, form in enumerate(cands):
                if filter_invalid and not is_valid_vn_or_clean_candidate(form, b_low):
                    continue
                norm_form = normalize_tone_placement(form.lower(), style=style)
                # Zipf-like deterministic weight fallback preserving original frequency ordering
                form_counts_by_base[b_low][norm_form] += max(1, (len(cands) - rank) * 10)

    new_bases: List[str] = []
    new_vocab: List[str] = []
    vid: Dict[str, int] = {}
    new_rows: List[List[int]] = []
    new_counts: List[List[float]] = []

    dropped_identity = 0
    total_original = len(bases)

    for b in sorted(form_counts_by_base):
        cnt = form_counts_by_base[b]
        if not cnt:
            continue
        top = cnt.most_common(kmax)
        if drop_identity_singletons and len(top) == 1 and top[0][0] == b:
            keep_vn_id = (
                min_identity_vn_count > 0
                and isinstance(ctab_or_counts, dict)
                and 1 <= len(b) <= 7
                and all(c in VN_BASE_LETTERS for c in b)
                and top[0][1] >= min_identity_vn_count
            )
            if not keep_vn_id:
                dropped_identity += 1
                continue
        new_bases.append(b)
        row_ids: List[int] = []
        row_cnts: List[float] = []
        for form, freq in top:
            if form not in vid:
                vid[form] = len(new_vocab)
                new_vocab.append(form)
            row_ids.append(vid[form])
            row_cnts.append(float(freq))
        while len(row_ids) < kmax:
            row_ids.append(-1)
            row_cnts.append(0.0)
        new_rows.append(row_ids)
        new_counts.append(row_cnts)

    stats = {
        "original_bases": total_original,
        "retained_bases": len(new_bases),
        "retained_vocab": len(new_vocab),
        "dropped_identity_singletons": dropped_identity,
        "reduction_pct": (1.0 - len(new_bases) / max(1, total_original)) * 100.0,
    }
    return new_bases, new_vocab, new_rows, new_counts, stats
