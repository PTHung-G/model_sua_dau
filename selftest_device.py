"""Guards against the 'index is on cpu, weights on cuda' class of bug.

This machine has no GPU, so instead of simulating CUDA we (a) run every batch
generator on CPU and assert all outputs share one device, and (b) statically
assert that every torch.tensor(...) call inside a batch generator passes an
explicit device. Both generators failed (b) before this check existed.
"""
import ast, json, torch
import evaluate as ev
import pilot_balance as pb

cand=json.load(open("data/cand.json")); forms=sorted(cand)
chars=sorted({c for s in forms for c in s}); c2i={c:i for i,c in enumerate(chars,start=1)}
nchar=len(chars)+2; max_c=max(len(v) for v in cand.values())
forms_id={s:i for i,s in enumerate(forms)}
sents=[l.split() for l in open("data/va.txt",encoding="utf-8")][:200]

def check_parity(name, batch):
    devs={t.device for t in batch}
    print(f"{name}: {len(batch)} tensors, devices={devs}")
    assert len(devs)==1, f"{name} mixes devices: {devs}"

check_parity("pilot batch", next(pb.make_batches(
    sents,cand,c2i,max_c,8,__import__("numpy").random.default_rng(0),
    nchar=nchar,forms=forms_id)))
check_parity("eval batch", next(ev.Tokens(sents,cand,c2i).batches(8,device="cpu")))

# static check: torch.tensor(...) must pass device=
for path, fn in (("pilot_balance.py","make_batches"), ("evaluate.py","batches")):
    tree=ast.parse(open(path).read())
    target=next(n for n in ast.walk(tree)
                if isinstance(n,ast.FunctionDef) and n.name==fn)
    bad=[n.lineno for n in ast.walk(target)
         if isinstance(n,ast.Call) and getattr(n.func,"attr","")=="tensor"
         and not any(k.arg=="device" for k in n.keywords)]
    print(f"{path}::{fn}  torch.tensor calls without device= : {bad}")
    assert not bad, f"{path}:{bad} creates tensors on the default device"

# catch the default-device trap: evaluate() defaults to cpu, so any call site
# evaluating a CUDA model must pass device= explicitly.
tree=ast.parse(open("pilot_balance.py").read())
calls=[n for n in ast.walk(tree)
       if isinstance(n,ast.Call) and getattr(n.func,"attr","")=="" 
       and getattr(getattr(n.func,"value",None),"id","")=="ev"]
calls+=[n for n in ast.walk(tree)
        if isinstance(n,ast.Call) and getattr(n.func,"attr","")=="evaluate"]
missing=[n.lineno for n in calls if not any(k.arg=="device" for k in n.keywords)]
print(f"pilot_balance.py: evaluate() calls without device= : {missing}")
assert not missing, f"evaluate() at lines {missing} would silently use the cpu default"

print("\nDEVICE CHECK PASSED")
