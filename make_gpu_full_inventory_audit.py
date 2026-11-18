"""Generate a private, hard-time-limited engineering GPU test. No long training."""
import base64,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
out=ROOT/'punct_gpu_full_inventory_kernel_v2';out.mkdir(exist_ok=False)
files=['charexp/sylmodel_punct.py','charexp/syltrain_punct.py','build_punct_tensors_v1.py','test_punct_alignment_audit.py','test_model_punct_regression.py']
payload={p:base64.b64encode((ROOT/p).read_bytes()).decode() for p in files}
worker='''import json,time,sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'charexp'))
import syltrain_punct as t
import sylmodel_punct as m
started=time.monotonic()
torch.set_num_threads(2)
assert torch.cuda.is_available(), 'GPU required'
device='cuda'
root=next(p.parent for p in Path('/kaggle/input').rglob('train_ctab.npy'))
assert json.loads((root/'manifest.json').read_text())['pipeline_schema']==2
train=t.load_data(root/'train');val=t.load_data(root/'validation')
nc,bc,cc,cid,_=t.build_tensors(str(root/'train'))
npunct=len(json.loads((root/'train_punct.json').read_text()))
torch.manual_seed(42)
model=m.PunctSylCRF(nc,bc,cc,cid,npunct).to(device)
starts,lengths=t.segments(train[2],train[3]);rng=np.random.default_rng(42)
# Confirm GPU scores do not change with batch padding (after activating gap projections).
model.eval()
with torch.no_grad():
 model.gap_proj.weight.normal_(0,.1)
 model.edge_gate.weight.normal_(0,.1)
 short=int(np.argmin(lengths));long=int(np.argmax(lengths))
 a=t.make_batch(train,starts,lengths,np.array([short]),device)
 b=t.make_batch(train,starts,lengths,np.array([short,long]),device)
 ea,ta,_=model(a[0],a[2],a[3],a[4]);eb,tb,_=model(b[0],b[2],b[3],b[4]);n=int(a[2][0])
 assert torch.allclose(ea[0,:n],eb[0,:n],atol=2e-4,rtol=2e-4),'GPU padding emissions mismatch'
 assert torch.allclose(ta[0,:n-1],tb[0,:n-1],atol=2e-4,rtol=2e-4),'GPU padding transitions mismatch'
opt=torch.optim.AdamW(model.parameters(),lr=.0015)
losses=[];step=0;model.train();training_started=time.monotonic()
while time.monotonic()-started<225 and time.monotonic()-training_started<120:
 for idx in t.epoch_batches(lengths,32,rng):
  if time.monotonic()-started>=225 or time.monotonic()-training_started>=120:break
  x,y,ln,ff,gg=t.make_batch(train,starts,lengths,idx,device)
  opt.zero_grad(set_to_none=True)
  e,tr,mask=model(x,ln,ff,gg);loss=m.nll(e,tr,mask,y,ln)
  assert torch.isfinite(loss),'nonfinite loss'
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
  losses.append(float(loss.detach()));step+=1
  if step%100==0:print('step',step,'loss',losses[-1],flush=True)
acc,tokens=t.evaluate(model,val,16)
assert step>0
result={'status':'PASS','purpose':'engineering test only, not model comparison','pipeline_schema':2,'gpu':torch.cuda.get_device_name(0),'optimizer_steps':step,'initial_20_loss_mean':float(np.mean(losses[:20])),'final_20_loss_mean':float(np.mean(losses[-20:])),'validation_covered_accuracy':acc,'validation_tokens':tokens,'padding_invariance_gpu':True,'finite_losses_gradients':True,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/1e6,'elapsed_seconds':time.monotonic()-started,'subset_inventory':False,'full_inventory_memory_verified':True,'inventory_bases':int(bc.shape[0]),'inventory_forms':int(cc.shape[0]),'batch_size':32,'training_seconds':time.monotonic()-training_started}
Path('/kaggle/working/gpu_smoke_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
'''
main="""import base64,json,subprocess,sys,time
from pathlib import Path
started=time.monotonic()
"""+f'payload={payload!r}\n'+"""for name,value in payload.items():
 target=Path('/kaggle/working',name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(base64.b64decode(value))
"""+f"Path('/kaggle/working/worker.py').write_text({worker!r})\n"+"""try:
 subprocess.run([sys.executable,'-m','unittest','test_punct_alignment_audit','test_model_punct_regression','-v'],check=True,timeout=35,cwd='/kaggle/working')
 subprocess.run([sys.executable,'/kaggle/working/worker.py'],check=True,timeout=max(1,270-(time.monotonic()-started)),cwd='/kaggle/working')
except Exception as exc:
 Path('/kaggle/working/gpu_smoke_failure.json').write_text(json.dumps({'status':'FAIL','error':str(exc),'elapsed_seconds':time.monotonic()-started}))
 raise
"""
(out/'main.py').write_text(main)
(out/'kernel-metadata.json').write_text(json.dumps({'id':'tunhngphm/vi-diac-full-inventory-gpu-check-v2','title':'Vi diac full inventory GPU check v2','code_file':'main.py','language':'python','kernel_type':'script','is_private':True,'enable_gpu':True,'enable_internet':False,'dataset_sources':['tunhngphm/vi-diac-full-inventory-audit-tensors-v2'],'competition_sources':[],'kernel_sources':[]},indent=2))
compile(main,'main.py','exec');compile(worker,'worker.py','exec')
print('Generated isolated GPU test: overall child execution cap 270s; optimizer window 180s.')
