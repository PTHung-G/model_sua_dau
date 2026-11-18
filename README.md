# model_sua_dau — ViDiCRF

Vietnamese diacritic restoration using a candidate-constrained neural CRF.
This repository is a curated source snapshot of the research workspace, not its 58 GB of experimental artifacts. Python 3.10–3.12 x64 is recommended. No GPU is required for the demo.

## Run the portable demo on Windows

```powershell
git clone https://github.com/PTHung-G/model_sua_dau.git
cd model_sua_dau
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy==2.2.6
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

**Model transfer is required:** copy `model.pt` from your previously downloaded DauDemo package into `windows_demo_build/DauDemo/model.pt`. Model weights are NOT included in Git. The vocabulary/inventory JSON files are included. See `artifacts.json` for expected sizes and SHA-256 hashes. There is no automatic model download.

If dependencies are already installed in this repository's `.venv`, skip all installation commands and run only:

```powershell
.\.venv\Scripts\python.exe .\windows_demo_build\DauDemo\server.py
```

The demo prints `Ready: http://127.0.0.1:<port>` and opens the browser. Keep the terminal open; Ctrl+C stops it. It listens only on loopback. No text is sent to an external API. `--no-browser` disables browser auto-opening.

On Linux, create a venv, install the same dependencies, then run:

```bash
.venv/bin/python windows_demo_build/DauDemo/server.py --no-browser
```

The inference bundle was tested on Linux; its Windows launcher has not been executed in this environment.

## Source layout

- `windows_demo_build/DauDemo/`: self-contained demo UI, HTTP server and inference pipeline. Best starting point for UI/inference changes.
- `charexp/sylmodel_punct.py`: neural architecture.
- `charexp/syltrain_punct.py`: training entry point.
- `build_punctuation_corpus_v1.py`, `build_punct_tensors_v1.py`: corpus/tensor preparation.
- `eval_punct_vs_best.py`: frozen-model restoration and evaluation.
- `proposal_d3/random_eval_20261115/`: sampling/audit and baseline evaluation scripts; raw evaluation paragraphs are not included.
- `proposal_d3/`, `proposal_d3_vi/`, `proposal_d1/`: LaTeX report sources and template files.
- Root-level Python scripts: historical research experiments. These are preserved for development, not all supported as standalone commands.

The portable demo includes its own model implementation. When modifying the training architecture, update the portable implementation deliberately and keep checkpoint compatibility in mind.

## Research setup and artifacts

`requirements.txt` describes demo dependencies. `requirements-research.txt` is a starting set for research scripts, not a fully locked environment. The original evaluation workspace used PyTorch 2.11 CPU; the portable Windows launcher specifies PyTorch 2.6.0 CPU. Raw datasets, prepared tensors, model checkpoints, logs, generated PDFs and notebook outputs are deliberately excluded. Some historical scripts contain original absolute paths; adapt those paths and supply their datasets before running. Cloning alone does not reproduce training.

The full-workspace `live_demo/server.py` expects additional checkpoint and inventory paths; use the portable server above for a fresh clone.

## Development from another machine

```bash
git pull --ff-only
git add <changed-source-files>
git commit -m "Describe the change"
git push origin main
```

Set up your own GitHub authentication on each machine. Never commit tokens, SSH private keys, `.venv`, or raw user input. No private credentials are included here.

## Limitations and data provenance

The frozen model completed two epochs and part of a third, not four completed epochs. The exploratory quality evaluation does not establish superiority over prior neural systems or isolate the CRF contribution. Real typing/OCR errors and unsupported candidate forms remain limitations.

Training data source: Wikimedia Wikipedia `20231101.vi`, https://huggingface.co/datasets/wikimedia/wikipedia . Source text is subject to its upstream licensing/attribution requirements. This repository does not redistribute the corpus and does not assign a new license to upstream text or the ICML template. No blanket open-source license is asserted for the project code without the authors' decision.
