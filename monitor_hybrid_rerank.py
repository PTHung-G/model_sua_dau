"""Monitor Kaggle hybrid reranker and download artifacts, including failures."""
from pathlib import Path
import datetime
import time
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
ref = 'tunhngphm/vi-diac-hybrid-rerank'
last = None
for i in range(360):
    try:
        status = api.kernels_status(ref)
        state = str(getattr(status, 'status', status)).upper()
        if state != last or i % 10 == 0:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), state, flush=True)
        last = state
        terminal = any(s in state for s in ('COMPLETE', 'ERROR', 'CANCEL'))
        if terminal:
            folder = 'kaggle_hybrid_rerank_output' if 'COMPLETE' in state else 'kaggle_hybrid_rerank_failure'
            out = Path(__file__).resolve().parent / folder
            out.mkdir(exist_ok=True)
            api.kernels_output(ref, path=str(out))
            print('DOWNLOADED', out, flush=True)
            if 'COMPLETE' not in state:
                raise SystemExit('Training failed; inspect downloaded logs/checkpoints')
            break
    except Exception as exc:
        print('MONITOR_RETRY', type(exc).__name__, str(exc)[:200], flush=True)
    time.sleep(120)
else:
    raise TimeoutError('Monitor exceeded 12h')
