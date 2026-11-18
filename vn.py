"""Vietnamese diacritic utilities (no external deps beyond unicodedata)."""
import unicodedata

# combining marks
TONE_MARKS = {
    "\u0300": "huyen",  # grave
    "\u0301": "sac",    # acute
    "\u0309": "hoi",    # hook above
    "\u0303": "nga",    # tilde
    "\u0323": "nang",   # dot below
}
TONE_CHARS = "".join(TONE_MARKS)

# vowel modifiers (quality, not tone)
MOD_MARKS = {"\u0306": "breve", "\u0302": "circumflex", "\u031b": "horn"}

MOD_LETTERS = {
    "ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u",
    "Ă": "A", "Â": "A", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U",
    "đ": "d", "Đ": "D",
}


def decompose(s):
    """Return (base_string, tones, mods) with all marks removed from base."""
    base, tones, mods = [], [], []
    for ch in unicodedata.normalize("NFD", s):
        if ch in TONE_MARKS:
            tones.append(TONE_MARKS[ch])
        elif ch in MOD_MARKS:
            mods.append(MOD_MARKS[ch])
        else:
            base.append(ch)
    return "".join(base), tones, mods


def strip_tone(s):
    """Remove tone marks only; keep vowel quality (ơ ư ă â ê ô đ)."""
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in unicodedata.normalize("NFD", s) if ch not in TONE_MARKS)
    )


def strip_all(s):
    """Remove tone marks AND vowel modifiers AND đ->d: full 'khong dau' form."""
    out = []
    for ch in unicodedata.normalize("NFD", s):
        if ch in TONE_MARKS or ch in MOD_MARKS:
            continue
        out.append(ch)
    s2 = unicodedata.normalize("NFC", "".join(out))
    return "".join(MOD_LETTERS.get(c, c) for c in s2)


def tone_of(s):
    """Return the tone name of a syllable ('' if none/ngang)."""
    _, tones, _ = decompose(s)
    return tones[0] if tones else ""


_BASE = "abcdeghiklmnopqrstuvxy".replace("f","").replace("j","").replace("w","").replace("z","")
_VOWELS = "aăâeêioôơuưy"
_TONES = "\u0300\u0301\u0309\u0303\u0323"
SYLLABLE_OK = set(_BASE)
for _v in _VOWELS:
    SYLLABLE_OK.add(_v)
    for _t in _TONES:
        SYLLABLE_OK.add(unicodedata.normalize("NFC", _v + _t))
SYLLABLE_OK.add("đ")
SYLLABLE_OK |= {c.upper() for c in list(SYLLABLE_OK)}


def is_vn_word(tok):
    return tok and all(c in SYLLABLE_OK for c in tok)


if __name__ == "__main__":
    tests = ["tôi", "đi", "học", "đường", "phở", "Việt", "Nam", "người", "được", "ơi", "mưa"]
    print(f"{'word':10s} {'strip_tone':12s} {'strip_all':10s} tone")
    for t in tests:
        print(f"{t:10s} {strip_tone(t):12s} {strip_all(t):10s} {tone_of(t):6s}")
    # roundtrip: strip_all must be idempotent and lossless w.r.t. our goal
    assert strip_all(strip_all("đường")) == strip_all("đường")
    print("\nroundtrip ok")
