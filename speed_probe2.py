import time, torch, pilot_arch as pa, pilot_balance as pb, context_pilot as cp
from pathlib import Path
payload = torch.load('kaggle_factorized_full_output/factorized_full_results/factorized_best.pt', map_location='cpu', weights_only=False)
va, _ = pa.reservoir(Path('data')/'va.txt', 500, cp.VALIDATION_SEED)
model = cp.model_from_checkpoint(payload, 'baseline').eval()
tables = (payload['tone_table'], payload['mod_table'])
batches = list(cp.raw_batches(va, payload, bs=1))
with torch.inference_mode():
    for _ in range(3):
        for _,_,_,t in batches[:20]:
            pa.candidate_logits(model, 'factorized', *t, tables)
    t0 = time.perf_counter()
    ntok = 0
    for _,_,rows,t in batches:
        pa.candidate_logits(model, 'factorized', *t, tables)
        ntok += sum(len(r) for r in rows)
    dt = time.perf_counter() - t0
print(dict(sentences=len(batches), tokens=ntok, seconds=round(dt,3),
           ms_per_sentence=round(dt/len(batches)*1000,3), ms_per_token=round(dt/ntok*1000,3),
           threads=torch.get_num_threads()))
