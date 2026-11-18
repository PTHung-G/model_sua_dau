"""Monitor sentence-loss pilot and retry transient download errors."""
from pathlib import Path
import datetime,time
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi();a.authenticate()
ref='tunhngphm/vi-diac-sentence-loss-pilot'
last=None
for i in range(180):
    try:
        status=a.kernels_status(ref);state=str(getattr(status,'status',status)).upper()
        if state!=last or i%10==0:print(datetime.datetime.now(datetime.timezone.utc).isoformat(),state,flush=True)
        last=state
        if 'COMPLETE' in state:
            p=Path('kaggle_sentence_loss_output').resolve();p.mkdir(exist_ok=True)
            a.kernels_output(ref,path=str(p));print('DOWNLOADED',p,flush=True);break
        if 'ERROR' in state or 'CANCEL' in state:
            p=Path('kaggle_sentence_loss_failure').resolve();p.mkdir(exist_ok=True)
            a.kernels_output(ref,path=str(p));raise SystemExit('Training failed; inspect logs')
    except Exception as e:print('MONITOR_RETRY',type(e).__name__,str(e)[:200],flush=True)
    time.sleep(120)
else:raise TimeoutError('Monitor exceeded6h')
