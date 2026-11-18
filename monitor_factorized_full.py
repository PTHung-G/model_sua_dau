"""Monitor full factorized Kaggle run and download outputs, retrying transient failures."""
import datetime
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

a = KaggleApi()
a.authenticate()
ref = 'tunhngphm/vi-diac-factorized-full-v2'
last = None
for attempt in range(360):
    try:
        status = a.kernels_status(ref)
        state = str(getattr(status, 'status', status)).upper()
        if state != last or attempt % 10 == 0:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), state, flush=True)
        last = state
        if 'COMPLETE' in state:
            root = Path('kaggle_factorized_full_output').resolve()
            root.mkdir(exist_ok=True)
            a.kernels_output(ref, path=str(root))
            print('DOWNLOADED', root, flush=True)
            break
        if 'ERROR' in state or 'CANCEL' in state:
            root = Path('kaggle_factorized_full_failure').resolve()
            root.mkdir(exist_ok=True)
            a.kernels_output(ref, path=str(root))
            raise SystemExit('Training terminal failure; inspect downloaded logs')
    except Exception as exc:
        print('RETRYABLE_MONITOR_ERROR',type(exc).__name__,str(exc)[:240],flush=True)
    time.sleep(120)
else:
    raise TimeoutError('Full training monitor exceeded twelve hours')
