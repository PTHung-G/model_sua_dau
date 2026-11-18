"""Bounded remote status monitor; download and verify only after completion."""
import datetime
import subprocess
import sys
import time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
ref = 'tunhngphm/vi-diac-architecture-pilot'
last = None
for attempt in range(180):
    try:
        status = api.kernels_status(ref)
        state = str(getattr(status, 'status', status))
        if state != last or attempt % 10 == 0:
            print(datetime.datetime.now(datetime.timezone.utc).isoformat(), state, flush=True)
        last = state
        if 'COMPLETE' in state.upper():
            out = Path('kaggle_arch_pilot_output').resolve()
            out.mkdir(exist_ok=True)
            api.kernels_output(ref, path=str(out))
            roots = list(out.rglob('results.json'))
            assert len(roots) == 1, roots
            subprocess.run([sys.executable, 'verify_pilot.py', str(roots[0].parent)], check=True)
            print('COMPLETE_DOWNLOADED_AND_VERIFIED', roots[0], flush=True)
            break
        if 'ERROR' in state.upper() or 'CANCEL' in state.upper():
            print('TERMINAL_FAILURE', status, flush=True)
            out = Path('kaggle_arch_pilot_failure').resolve()
            out.mkdir(exist_ok=True)
            api.kernels_output(ref, path=str(out))
            sys.exit(1)
    except Exception as exc:
        print('MONITOR_EXCEPTION', type(exc).__name__, str(exc), flush=True)
        if last and 'COMPLETE' in last.upper():
            raise
    time.sleep(120)
else:
    raise TimeoutError('Remote pilot did not finish within six-hour monitor window')
