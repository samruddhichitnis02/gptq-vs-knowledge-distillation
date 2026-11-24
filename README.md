# DATA 266 — Homework 12: GPTQ quantisation and knowledge distillation

**Parameters:** `SID4 = 2122`, `SEED = 2122`, `HP_ID = 2122 mod 6 = 4`. HW12 defines no HP_ID-specific setting, and the PDF says
HP_ID does not add another temperature.
**Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) under WSL2 Ubuntu. Every model in a comparison ran on this GPU with the same software:
torch 2.15.0 (CUDA 13.2), transformers 5.24.1, GPT-QModel 7.12.2, optimum 2.3.0, datasets 5.2.0.

| File | Contents |
|---|---|
| `HW12_Part1_GPTQ.ipynb` / `.html` | Part 1: OPT-125m base vs GPTQ 4-bit on WikiText-2 (code, outputs, findings) |
| `HW12_Part2_Distillation.ipynb` / `.html` | Part 2: BERT → DistilBERT distillation on AG News, both sweeps, per-class analysis, the quantisation-vs-distillation comparison |
| `metrics.md` | All reported numbers in one place |
| `RUN_LOG.txt` | Console output from the runs used for the reported results |
| `AI_USE.md` | AI-use disclosure |
| `results/` | CSV/JSON metrics (`part1/summary.json`, `part2/summary.json`), per-run histories (`part2/runs/`), figures |
| `logs/` | Timestamped run logs and nbconvert execution logs |
| `requirements.txt` | Package versions used |

Model weights (`models/`, `teacher_model/`, `student_model_best/`, `checkpoints/`) are produced by the notebooks. They are
git-ignored because of their size.

## Part 1 — GPTQ (`facebook/opt-125m`, revision `27dcfa74d334bc871f3234de431e71c6eeba5dd6`)
4-bit, group size 128, calibrated on WikiText-2 train (46.5 s). Evaluated on WikiText-2 test: perplexity over 140 non-overlapping
2048-token windows.

**Table HW11.1 — GPTQ comparison on the same LLM/task**

| Configuration | Quality metric (PPL / eval loss) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 27.66 / 3.3198 | 2095.9 / 2426.9 | 1.265 | 242.3 |
| GPTQ-quantized LLM (4-bit) | 32.19 / 3.4718 | 4136.7 / 4858.1 | 1.148 | 122.7 |

Latency is for batch 1, 128 generated tokens. GPTQ halves the on-disk size and resident weights (embeddings stay fp16) and costs
+4.5 perplexity. On this GPU it is about 2× slower at batch 1, because a 125M model is bound by kernel launches rather than by weight bandwidth.

## Part 2 — Knowledge distillation (AG News)
The teacher is a **stand-in** `bert-base-uncased` fine-tuned on AG News: the instructor's Drive `teacher_model` link was
access-restricted (HTTP 401). The notebook loads `./teacher_model` if present, so the official teacher can be dropped in and the notebook re-run.
The loss is `α·CE + (1−α)·T²·KL`. Model selection uses a 6,000-sample validation split held out from the training set.

| Sweep | Results (val / test accuracy) | Best |
|---|---|---|
| T sweep, α = 0.5 | T=1 94.40/94.46 · T=2 94.35/94.38 · T=5 94.33/94.34 · T=10 94.17/94.36 | **T = 1** |
| α sweep, T = 1 | α=0.25 94.23/94.36 · α=0.5 94.40/94.46 · α=0.75 94.48/94.32 | **α = 0.75** (by val) |

**Table HW11.2 — the separate distillation comparison on AG News**

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT-base) | 94.20 % | 13.87 / 21.85 | 0.929 | 418.4 |
| Best distilled student (T=1, α=0.75) | 94.32 % | 7.82 / 11.73 | 0.771 | 256.1 |

The student keeps the teacher's accuracy, with 1.77× lower latency, 2× the throughput and 0.61× the size.
**World** is the only class that degrades, by just −0.21 pp. The items the student misses are World-labelled stories whose
vocabulary belongs to another class (airlines, NFL, rugby, election law).

## Quantisation vs. distillation
GPTQ shrinks bytes per weight without training but leaves the compute graph unchanged. Distillation removes layers, and therefore FLOPs.
For a **latency-bound** deployment I would try **distillation** first; for a **memory-bound** deployment I would try **GPTQ** first.
The full 5-sentence comparison is at the end of the Part 2 notebook and in `metrics.md`.

## Reproduce
```bash
python3 -m venv ~/hw12 && source ~/hw12/bin/activate
pip install -r requirements.txt          # on Linux/WSL; GPT-QModel needs a C/C++ toolchain
python -m ipykernel install --user --name hw12
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hw12 HW12_Part1_GPTQ.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=hw12 --ExecutePreprocessor.timeout=14400 HW12_Part2_Distillation.ipynb
```
`HW12_QUICK=1` runs Part 2 on small subsets as a smoke test, writing to `*_quick` folders.
