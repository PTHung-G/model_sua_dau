"""Eval-mode learning diagnostics on identical fixed rows across checkpoints."""
import torch
import torch.nn.functional as F
import sentence_loss_epoch as se
import sentence_loss_pilot as sl

@torch.inference_mode()
def measure(model,payload,train_rows,val_rows):
    was=model.training;model.eval();output={}
    forms={s:i for i,s in enumerate(payload['forms'])}
    device=next(model.parameters()).device
    try:
        for split,rows in [('train',train_rows),('validation',val_rows)]:
            n=0;ce=0.;correct=0;base_ce=0.;base_correct=0
            for start in range(0,len(rows),64):
                batch,_=se.collate(rows[start:start+64],payload,device=device,forms=forms)
                if batch is None:continue
                xc,cl,f,y,ln,mask,ok=batch
                if not (mask&ok).any():continue
                target=sl.training_target(mask,ok,ln)
                scores=model.logits(xc,cl,f,ln,mask);_,base=model.frozen(xc,cl,f,ln,mask)
                ce+=float(F.cross_entropy(scores[target],y[target],reduction='sum'))
                base_ce+=float(F.cross_entropy(base[target],y[target],reduction='sum'))
                n+=int(target.sum());correct+=int((scores.argmax(-1)[target]==y[target]).sum())
                base_correct+=int((base.argmax(-1)[target]==y[target]).sum())
            output[split]=dict(tokens=n,ce=ce/max(n,1),accuracy=correct/max(n,1),
                               base_ce=base_ce/max(n,1),base_accuracy=base_correct/max(n,1))
        return output
    finally:model.train(was)
