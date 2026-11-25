# Matched unary-scoring fine-tune

This exploratory experiment is separate from the D3 report. Both arms fine-tune the entire punctuation-aware ViDiCRF from the same checkpoint. They add parameter-matched flat and letter–tone corrections to the original, still-trainable unary at scale 1.0. Both start with identical predictions. The candidate lattice and decoding algorithm are held fixed. The first 480 batches reproduce the short pilot; subsequent batches use fresh segments from disjoint sampled rows. No test access.

The fivefold Kaggle run completed on Tesla T4; saved final weights reproduced the dev metrics on reload. On the same 1,000 covered dev paragraphs (40,174 tokens), the pretrained baseline scored 38,787 correct tokens and made 777 tone-only errors.

| Steps | Cumulative train tokens per arm | Correct tokens: flat / letter–tone | Tone-only errors: flat / letter–tone |
|---:|---:|---:|---:|
| 480 | 250,879 | 38,836 / 38,829 | 751 / 750 |
| 1,200 | 626,082 | 38,873 / 38,879 | 715 / 715 |
| 2,400 | 1,248,417 | 38,898 / 38,904 | 698 / 696 |

At step 2,400, letter–tone is ahead by six tokens (+0.015 percentage points; paired document bootstrap 95% interval [−0.031, +0.056] points). Exact paragraphs are 652/1,000 for flat and 650/1,000 for letter–tone. Both arms improve with the larger budget, but this one-seed comparison on reused dev does not establish a scorer advantage; it increases both data exposure and number of updates.

## Reproduce

Run `python -m unittest -v test_punct_unary_pilot` for the scorer tests. Run `python build_punct_unary_finetune_kaggle.py <new-package-dir> --fivefold` to create a private GPU script package, then `kaggle kernels push -p <new-package-dir> --accelerator NvidiaTeslaT4 --timeout 5400` using an account with access to the verified tensor dataset and pretrained checkpoint. The builder targets the existing experiment slug; change the target in the generated metadata if a separate notebook is desired. Verify the resulting manifest, per-step metrics and checkpoint hashes before interpreting scores. Datasets, checkpoints, Kaggle outputs and credentials are not included in this repository update.
