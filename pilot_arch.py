"""Matched baseline / factorized output / dual-attention pilot (no combined arm).

Run: python pilot_arch.py --arms baseline,factorized,dual
All selection/reporting uses validation only; test data are never opened.
Factorization uses lossless POSITIONAL tone and modifier signatures, rather than
six tone categories: the frozen dictionary contains e.g. ái / aí and foreign
multi-tone forms. Both vocabularies are dynamic; no candidates are rewritten.
It FAILS and logs if a refined pair still collides within a stem.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import evaluate as ev
import pilot_balance as pb
from attn_model import AttnDiac
from vn import strip_all, strip_tone, tone_of
import unicodedata
from vn import TONE_MARKS

TONES = ("", "huyen", "sac", "hoi", "nga", "nang")
ARMS = ("baseline", "factorized", "dual")


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def reservoir(path, count, seed):
    """Algorithm R on raw lines: O(count) storage, never load the whole corpus."""
    rng = random.Random(seed)
    selected = []
    source_hash = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as fh:
        for index, raw in enumerate(fh):
            source_hash.update(raw)
            total += 1
            if len(selected) < count:
                selected.append((index, raw))
            else:
                j = rng.randrange(index + 1)
                if j < count:
                    selected[j] = (index, raw)
    selected.sort(key=lambda pair: pair[0])
    sample_hash = hashlib.sha256()
    entries, sents = [], []
    for index, raw in selected:
        sample_hash.update(raw)
        entries.append({"line_1based": index + 1,
                        "sha256": hashlib.sha256(raw).hexdigest()})
        sents.append(raw.decode("utf-8").split())
    return sents, {"source": str(Path(path).resolve()), "source_lines": total,
                   "source_sha256": source_hash.hexdigest(), "seed": seed,
                   "requested": count, "sampled": len(selected),
                   "sample_sha256": sample_hash.hexdigest(), "entries": entries}


def modifier_signature(word):
    """Retain POSITION and modified character (including the stroke in đ)."""
    plain, modified = strip_all(word), strip_tone(word)
    if len(plain) != len(modified):
        raise ValueError(f"Unaligned modifier signature: {word!r}")
    return tuple((i, char) for i, (char, base) in enumerate(zip(modified, plain))
                 if char != base)


def tone_signature(word):
    """Lossless positional tone marks, including multiplicity in foreign words.

    Six tone categories alone collide (e.g. ái / aí); the parent-approved
    lossless refinement retains tone position rather than filtering candidates.
    """
    position = -1
    marks = []
    for char in unicodedata.normalize("NFD", word):
        if char in TONE_MARKS:
            marks.append((position, char))
        elif not unicodedata.combining(char):
            position += 1
    return tuple(marks)


def factor_tables(cand, forms, collision_path):
    """Build global signature vocabulary from the ENTIRE frozen dictionary."""
    tone_signatures = sorted({tone_signature(w) for s in forms for w in cand[s]})
    tone_ids = {s: i for i, s in enumerate(tone_signatures)}
    signatures = sorted({modifier_signature(w) for s in forms for w in cand[s]})
    signature_ids = {s: i for i, s in enumerate(signatures)}
    max_c = max(map(len, cand.values()))
    tone_table = torch.zeros((len(forms), max_c), dtype=torch.long)
    mod_table = torch.zeros_like(tone_table)
    collisions = []
    for fi, stem in enumerate(forms):
        seen = {}
        for ci, word in enumerate(cand[stem]):
            signature = modifier_signature(word)
            tone = tone_signature(word)
            if strip_all(word) != stem:
                raise ValueError(f"Invalid frozen candidate {stem!r}: {word!r}")
            pair = (tone_ids[tone], signature_ids[signature])
            if pair in seen:
                collisions.append({"stem": stem, "first": seen[pair],
                                   "second": word, "tone": tone,
                                   "signature": signature})
            seen[pair] = word
            tone_table[fi, ci], mod_table[fi, ci] = pair
    atomic_json(collision_path, {"n_collisions": len(collisions),
                                "n_signatures": len(signatures),
                                "n_tone_signatures": len(tone_signatures),
                                "factorization": "lossless positional tones and modifiers",
                                "collisions": collisions})
    if collisions:
        raise ValueError(f"Non-injective factorization: {len(collisions)} collisions; "
                         f"see {collision_path}. Frozen candidates NOT modified.")
    return {"modifier": signatures, "tone": tone_signatures}, tone_table, mod_table


class FactorizedDiac(AttnDiac):
    """Flat encoder, with dynamic modifier head; never use legacy n_mod=7."""
    def __init__(self, nchar, max_c, n_mod, n_tone, **kwargs):
        super().__init__(nchar, max_c, factorized=False, **kwargs)
        self.out = nn.Identity()
        self.tone_head = nn.Linear(kwargs["dim"], n_tone)
        self.mod_head = nn.Linear(kwargs["dim"], n_mod)
        for head in (self.tone_head, self.mod_head):
            nn.init.normal_(head.weight, std=kwargs["out_std"])
            nn.init.zeros_(head.bias)

    def forward(self, xc, cl, mask=None):
        hidden = super().forward(xc, cl, mask=mask)
        return self.tone_head(hidden), self.mod_head(hidden)


def candidate_logits(model, arm, xc, cl, f, ln, mask, tables):
    output = model.forward(xc, cl, mask=mask)
    if arm == "factorized":
        tone, modifier = output
        tone_table, mod_table = tables
        logits = tone.gather(-1, tone_table[f]) + modifier.gather(-1, mod_table[f])
    else:
        logits = output
    slots = torch.arange(logits.shape[-1], device=logits.device)
    return logits.masked_fill(slots >= ln[..., None], ev.NEG)


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_model(model, arm, tokens, tables):
    was_training = model.training
    model.eval()
    synchronize()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            def predict(xc, cl, f, ln):
                # ev's API omits mk. Its padding character is unique, including
                # for single-candidate tokens and genuine form id zero.
                mask = xc[..., 0] != tokens.pad_c
                return candidate_logits(model, arm, xc, cl, f, ln, mask, tables).argmax(-1)
            result = ev.evaluate(predict, tokens, device=pb.DEV, bs=128)
        synchronize()
        return result, time.perf_counter() - start
    finally:
        model.train(was_training)


def checkpoint(path, model, opt, scheduler, rng, step, metadata, marks):
    synchronize()
    payload = dict(metadata, step=step, marks=marks,
                   model=model.state_dict(), optimizer=opt.state_dict(),
                   scheduler=scheduler.state_dict(),
                   rng={"python": random.getstate(), "numpy": np.random.get_state(),
                        "batch_generator": rng.bit_generator.state,
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train_arm(arm, args, tr, tokens, cand, char2id, forms, signatures, tables,
              manifests, outdir):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    config = dict(syl=192, dim=384, blocks=1, heads=8, ch_dim=16,
                  window="dual" if arm == "dual" else 8, out_std=0.1,
                  drop=0.0, ff_mult=1, packed_chars=True)
    if "packed_chars" not in inspect.signature(AttnDiac.__init__).parameters:
        raise RuntimeError("This pilot requires AttnDiac(packed_chars=True) to avoid "
                           "batch-padding-dependent character representations.")
    max_c = max(map(len, cand.values()))
    nchar = len(char2id) + 2
    if arm == "factorized":
        model = FactorizedDiac(nchar, max_c, len(signatures["modifier"]),
                               len(signatures["tone"]), **config)
    else:
        model = AttnDiac(nchar, max_c, factorized=False, **config)
    model = model.to(pb.DEV)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    rng = np.random.default_rng(args.seed)
    # Reconstructed identically per arm: same sentences, RNG, order and bs64.
    gen = pb.make_batches(tr, cand, char2id, max_c, 64, rng,
                          nchar=nchar, forms={s: i for i, s in enumerate(forms)})
    metadata = {"arm": arm, "seed": args.seed, "config": config,
                "run_config": vars(args), "params": nparam,
                "cand": cand, "forms": forms, "char2id": char2id,
                "max_c": max_c, "nchar": nchar,
                "signatures": signatures, "tones": TONES,
                "tone_table": tables[0].cpu() if tables else None,
                "mod_table": tables[1].cpu() if tables else None,
                "manifests": manifests,
                "objective": "candidate-masked CE: ambiguous covered tokens, fallback covered",
                "eval_batch_size": 128, "train_batch_size": 64}
    marks = {}
    train_seconds = eval_seconds = save_seconds = 0.0
    gate_sum = gate_count = 0
    active_mask = None
    handles = []
    if arm == "dual":
        def gate_hook(module, inputs, output):
            nonlocal gate_sum, gate_count
            if module.training and active_mask is not None:
                values = output.detach().sigmoid().squeeze(-1)[active_mask]
                gate_sum += float(values.sum())
                gate_count += values.numel()
        for block in model.blocks:
            handles.append(block.att.gate.register_forward_hook(gate_hook))
    wall_start = time.perf_counter()
    model.train()
    try:
        for step in range(1, args.steps + 1):
            synchronize()
            start = time.perf_counter()
            xc, cl, f, y, ln, mk, ok = next(gen)
            active_mask = mk
            logits = candidate_logits(model, arm, xc, cl, f, ln, mk, tables)
            ambiguous = mk & (ln > 1) & ok
            target = ambiguous if ambiguous.any() else (mk & ok)
            if not target.any():
                raise RuntimeError("Training batch has no covered labels")
            loss = F.cross_entropy(logits[target], y[target])
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at {arm} step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            synchronize()
            train_seconds += time.perf_counter() - start
            if step % 1000 == 0 or step == args.steps:
                result, seconds = evaluate_model(model, arm, tokens, tables)
                eval_seconds += seconds
                marks[step] = {"validation": result, "loss": float(loss.detach()),
                               "train_seconds": train_seconds, "eval_seconds": seconds}
                print(f"{arm} step={step} {ev.fmt('validation', result)} "
                      f"train_s={train_seconds:.2f} eval_s={seconds:.2f}", flush=True)
            if step % 500 == 0 or step == args.steps:
                start = time.perf_counter()
                metadata["timing"] = {"train_seconds": train_seconds,
                                      "eval_seconds": eval_seconds,
                                      "save_seconds_before_this_save": save_seconds}
                checkpoint(outdir / f"{arm}_s{args.seed}_latest.pt", model, opt,
                           scheduler, rng, step, metadata, marks)
                save_seconds += time.perf_counter() - start
    finally:
        for handle in handles:
            handle.remove()
    return {"arm": arm, "seed": args.seed, "params": nparam,
            "steps": args.steps, "validation": marks[args.steps]["validation"],
            "marks": marks, "train_seconds": train_seconds,
            "eval_seconds": eval_seconds, "save_seconds": save_seconds,
            "wall_seconds": time.perf_counter() - wall_start,
            "gate_local_mean": gate_sum / gate_count if gate_count else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--train-sents", type=int, default=100000)
    parser.add_argument("--val-sents", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=1729)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"))
    parser.add_argument("--out-dir", default="pilot_arch_results")
    parser.add_argument("--smoke", action="store_true",
                        help="cap steps at 2, train/val sample sizes at 128/32; architecture unchanged")
    args = parser.parse_args()
    arms = args.arms.split(",")
    if not arms or any(a not in ARMS for a in arms) or len(set(arms)) != len(arms):
        parser.error("--arms must contain distinct baseline,factorized,dual names")
    if min(args.steps, args.train_sents, args.val_sents) < 1:
        parser.error("steps and sample sizes must be positive")
    if args.smoke:
        args.steps = min(args.steps, 2)
        args.train_sents = min(args.train_sents, 128)
        args.val_sents = min(args.val_sents, 32)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    cand_raw = (data_dir / "cand.json").read_bytes()
    cand = json.loads(cand_raw)
    if not cand or any(not isinstance(cs, list) or not cs for cs in cand.values()):
        raise ValueError("Candidate dictionary must map stems to nonempty ordered lists")
    forms = sorted(cand)
    chars = sorted({c for s in forms for c in s})
    char2id = {c: i for i, c in enumerate(chars, 1)}
    signatures, tables = [], None
    # Validate before expensive sampling/training. Log even empty collision lists.
    if "factorized" in arms:
        signatures, tone_table, mod_table = factor_tables(
            cand, forms, outdir / "factorization_collisions.json")
        tables = (tone_table.to(pb.DEV), mod_table.to(pb.DEV))
    tr, tr_manifest = reservoir(data_dir / "tr.txt", args.train_sents, args.sample_seed)
    va, va_manifest = reservoir(data_dir / "va.txt", args.val_sents, args.sample_seed + 1)
    manifests = {"train": tr_manifest, "validation": va_manifest,
                 "cand_sha256": hashlib.sha256(cand_raw).hexdigest()}
    atomic_json(outdir / "sample_manifests.json", manifests)
    tokens = ev.Tokens(va, cand, char2id)
    if not tokens.n_tok:
        raise ValueError("Validation sample contains no evaluable tokens")
    synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        pick0 = ev.evaluate(lambda xc, cl, f, ln: torch.zeros_like(f), tokens,
                            device=pb.DEV, bs=128)
    synchronize()
    results = {"config": vars(args), "device": pb.DEV,
               "pick0_validation": pick0,
               "pick0_eval_seconds": time.perf_counter() - start,
               "n_val_tokens": tokens.n_tok, "arms": []}
    atomic_json(outdir / "results.json", results)
    print(ev.fmt("pick0 validation", pick0), flush=True)
    for arm in arms:
        result = train_arm(arm, args, tr, tokens, cand, char2id, forms,
                           signatures, tables, manifests, outdir)
        results["arms"].append(result)
        atomic_json(outdir / "results.json", results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
