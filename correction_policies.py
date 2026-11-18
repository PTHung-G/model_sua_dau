"""Pure numpy conservative policies and all-token metrics."""
import numpy as np

ALPHAS=(0.,.1,.25,.5,.75,1.)
THRESHOLDS=(0.,.05,.1,.2,.4)


def predict(data,kind,value):
    b=data['base'];r=data['rerank']
    if kind=='alpha':return (b+value*(r-b)).argmax(-1)
    if kind!='threshold':raise ValueError(kind)
    before=b.argmax(-1);after=r.argmax(-1)
    probs=np.exp(r-r.max(-1,keepdims=True));probs/=probs.sum(-1,keepdims=True)
    row=np.arange(len(b));gap=probs[row,after]-probs[row,before]
    return np.where(gap>=value,after,before)


def metrics(data,picks,n_sentences):
    gold=data['gold'];sid=data['sentence'];before=data['base'].argmax(-1)
    wrong=picks!=gold;base_wrong=before!=gold
    errs=np.bincount(sid,weights=wrong,minlength=n_sentences)
    base_errs=np.bincount(sid,weights=base_wrong,minlength=n_sentences)
    return dict(token_accuracy=float((~wrong).mean()),sentence_exact=float((errs==0).mean()),
                fixed_tokens=int((base_wrong&~wrong).sum()),broken_tokens=int((~base_wrong&wrong).sum()),
                changed_tokens=int((picks!=before).sum()),
                wrong_to_wrong=int(((picks!=before)&wrong&base_wrong).sum()),
                rescued_sentences=int(((base_errs>0)&(errs==0)).sum()),
                broken_sentences=int(((base_errs==0)&(errs>0)).sum()))


def select(data,kind):
    grid=ALPHAS if kind=='alpha' else THRESHOLDS
    # Higher accuracy, then more conservative settings.
    scored=[(float((predict(data,kind,x)==data['gold']).mean()),-x if kind=='alpha' else x,x) for x in grid]
    return max(scored)[2]
