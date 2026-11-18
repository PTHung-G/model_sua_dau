"""Features computed from stripped INPUT only. Zero is padding."""
import re
TOK = re.compile(r"[^\W_]+", re.UNICODE)
def case_id(s):
    if s.islower(): return 1
    if s.isupper(): return 3
    if s.istitle(): return 2
    return 4

def boundary_id(gap, first=False):
    if any(c in gap for c in ".!?…\n"): return 3
    if "," in gap: return 2
    if gap.strip(): return 4
    return 5 if first else 1

def features(source):
    matches=list(TOK.finditer(source)); result=[]; end=0
    for i,m in enumerate(matches):
        result.append((case_id(m.group()),boundary_id(source[end:m.start()],i==0)))
        end=m.end()
    return result
