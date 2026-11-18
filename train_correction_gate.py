"""Small frozen-score logistic gate; safe epoch-zero comparator and full checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from correction_policies import metrics

ROOT=Path(__file__).resolve().parent


def features(d):
    b=d['base'];r=d['rerank'];sid=d['sentence'];n=int(sid.max())+1
    def prob(x):
        p=np.exp(x-x.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)
    bp=prob(b);rp=prob(r);bi=b.argmax(-1);ri=r.argmax(-1);idx=np.arange(len(b))
    bs=np.sort(bp,axis=-1);rs=np.sort(rp,axis=-1)
    count=np.bincount(sid,minlength=n)
    mean=np.bincount(sid,weights=bp.max(-1),minlength=n)/np.maximum(1,count)
    uncertain=np.bincount(sid,weights=bp.max(-1)<.5,minlength=n)/np.maximum(1,count)
    x=np.stack([bp.max(-1),rp.max(-1),bs[:,-1]-bs[:,-2],rs[:,-1]-rs[:,-2],
                rp[idx,ri]-rp[idx,bi],bp[idx,bi]-bp[idx,ri],np.log(d['counts']),
                d['length']/40,d['position']/(d['length']+1),mean[sid],uncertain[sid]],axis=1)
    assert np.isfinite(x).all()
    return x.astype('float32')


def gate_predict(d,checkpoint):
    before=d['base'].argmax(-1);after=d['rerank'].argmax(-1)
    if checkpoint['epoch']==0:return after
    x=(features(d)-checkpoint['mean'])/checkpoint['std']
    score=x@checkpoint['model']['weight'].numpy().reshape(-1)+float(checkpoint['model']['bias'][0])
    return np.where(score>=0,after,before)


def main():
    torch.set_num_threads(2);torch.manual_seed(2403)
    out=ROOT/'correction_gate';out.mkdir(exist_ok=True)
    assert not (out/'gate_latest.pt').exists(),'Never overwrite a prior run'
    d=dict(np.load(ROOT/'correction_calibration/calibration_scores.npz'))
    order=np.random.default_rng(2403).permutation(512);fit_ids=order[:384];dev_ids=order[384:]
    fit=np.isin(d['sentence'],fit_ids);dev=np.isin(d['sentence'],dev_ids)
    before=d['base'].argmax(-1);after=d['rerank'].argmax(-1);gold=d['gold']
    useful=(before!=after)&((before==gold)^(after==gold))
    training=fit&useful
    assert training.sum()>10 and len(np.unique((after==gold)[training]))==2
    x=features(d);mean=x[fit].mean(0);std=np.maximum(x[fit].std(0),1e-4)
    X=torch.tensor((x[training]-mean)/std);Y=torch.tensor((after==gold)[training],dtype=torch.float32)
    model=torch.nn.Linear(x.shape[1],1);torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.1)
    def key(p):return (int((p[dev]==gold[dev]).sum()),-int((p[dev]!=after[dev]).sum()))
    best=key(after);history=[]
    def save(epoch,path):
        torch.save(dict(epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),
                        mean=mean,std=std,seed=2403,fit_sentence_ids=fit_ids,dev_sentence_ids=dev_ids,
                        training_examples=int(training.sum()),config={'lr':.01,'weight_decay':.1,'epochs':200},
                        history=history),path)
    save(0,out/'gate_best.pt');save(0,out/'gate_latest.pt')
    for epoch in range(1,201):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(X).flatten(),Y)
        loss.backward();opt.step()
        checkpoint=dict(epoch=epoch,model=model.state_dict(),mean=mean,std=std)
        picks=gate_predict(d,checkpoint);score=key(picks)
        history.append(dict(epoch=epoch,loss=float(loss.detach()),dev_correct=score[0],dev_rejections=-score[1]))
        save(epoch,out/'gate_latest.pt')
        if score>best:best=score;save(epoch,out/'gate_best.pt')
    for kind in ('best','latest'):
        checkpoint=torch.load(out/f'gate_{kind}.pt',weights_only=False)
        reloaded=torch.nn.Linear(11,1);reloaded.load_state_dict(checkpoint['model'],strict=True)
        assert np.isfinite(gate_predict(d,checkpoint)).all()
        print(kind,'epoch',checkpoint['epoch'],'fit decisive changes',int(training.sum()),'dev key',key(gate_predict(d,checkpoint)))
    (out/'fit_history.json').write_text(json.dumps(history,indent=2))

if __name__=='__main__':main()
