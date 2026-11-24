# Homework 12 — Metrics and Analysis

```text
SID4: 2122    SEED: 2122    HP_ID: 4 (no HP_ID-specific setting in HW12)
GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB), WSL2 Ubuntu
torch 2.15.0+cu132 · transformers 5.24.1 · GPTQModel 7.12.2 · optimum 2.3.0 · datasets 5.2.0
```

As item 1 of the PDF requires, the two parts are reported separately: they use different models and tasks.

---

## Part 1 — GPTQ on `facebook/opt-125m`

| Setting | Value |
|---|---|
| Model ID / revision | `facebook/opt-125m` @ `27dcfa74d334bc871f3234de431e71c6eeba5dd6` (same for both runs) |
| GPTQ config | 4-bit, group size 128, `desc_act=False`, symmetric, `dataset="wikitext2"` (train split) |
| Quantisation time / peak memory | 46.5 s / 1.02 GB |
| Inference kernel | GPT-QModel `MarlinLinear` (72 decoder linear layers) |
| Quality metric | WikiText-2 (`wikitext-2-raw-v1`) **test** perplexity, 140 non-overlapping 2048-token windows (287,645 tokens) |
| Latency | batch 1, 32-token prompt → 128 new greedy tokens, 5 warm-up + 30 timed requests |

**Table HW11.1 — GPTQ comparison on the same LLM/task**

| Configuration | Quality metric (PPL / eval loss) | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Base LLM (fp16) | 27.66 / 3.3198 | 2095.9 / 2426.9 | 1.265 | 242.3 |
| GPTQ-quantized LLM (4-bit) | 32.19 / 3.4718 | 4136.7 / 4858.1 | 1.148 | 122.7 |

Supporting measurements:

| Metric | Base fp16 | GPTQ 4-bit | GPTQ / base |
|---|---|---|---|
| Generation throughput (tok/s, from the p50) | 61.1 | 30.9 | 0.51× |
| 512-token prefill p50 / p95 (ms) | 17.95 / 22.79 | 27.75 / 30.98 | 1.55× (p50) |
| Resident weights after load (GB) | 0.233 | 0.116 | 0.50× |
| Weight files (MB) | 238.9 | 119.3 | 0.50× |

**Findings**
* Perplexity rises by 4.54 (+16 %). The base value matches the published 27.65.
* Size falls 2×, not 4×, because the token and positional embeddings and the LayerNorms stay fp16.
* Peak memory is dominated by activations (the 2048 × 50,272 logits), so it falls only 9 %.
* At batch 1 a 125M-parameter model is bound by kernel launches and Python overhead, not by weight bandwidth. The extra dequantisation work therefore makes GPTQ about 2× slower here.

---

## Part 2 — Knowledge distillation on AG News

| Setting | Value |
|---|---|
| Data | AG News: 120,000 train (114,000 fit + 6,000 stratified validation, `random_state=SEED`), 7,600 test |
| Teacher | **Stand-in** `bert-base-uncased` fine-tuned 1 epoch on the fit split (the instructor's Drive folder returned HTTP 401). Val 94.42 %, test 94.20 % |
| Teacher logits | extracted for all 120,000 training samples (177 s) |
| Student | `distilbert-base-uncased`, re-loaded and re-seeded (`SEED`) for every run |
| Fixed settings | 1 epoch, batch 32, AdamW lr 5e-5, wd 0.01, 6 % warm-up + linear decay, clip 1.0, fp16 autocast, max length 128, same data order |
| Loss | `α·CE(student, y) + (1−α)·T²·KL(softmax(t/T) ‖ softmax(s/T))` |

### Temperature sweep (α = 0.5)

| T | Val acc | Test acc | Test agreement with teacher | Train time (s) |
|---|---|---|---|---|
| **1** | **94.40 %** | **94.46 %** | 98.33 % | 327 |
| 2 | 94.35 % | 94.38 % | 98.28 % | 328 |
| 5 | 94.33 % | 94.34 % | 98.25 % | 329 |
| 10 | 94.17 % | 94.36 % | 98.39 % | 330 |

**Best T = 1** on validation. The 0.23 pp spread is within about one standard error (≈ 0.30 pp at n = 6,000).
The teacher's training logits are confident (mean max-probability 0.956 at T = 1), and higher T flattens them to near-uniform
(0.38 at T = 10). With only 4 classes there is little dark knowledge to transfer.

### Loss-weighting sweep (T = 1)

| α | Val acc | Test acc | Test agreement with teacher |
|---|---|---|---|
| 0.25 | 94.23 % | 94.36 % | 98.39 % |
| 0.50 | 94.40 % | 94.46 % | 98.33 % |
| **0.75** | **94.48 %** | 94.32 % | 97.86 % |

The best on validation is α = 0.75, and this student is used below. α = 0.5 is best on test, so the three are statistically indistinguishable.
More weight on the KD term increases agreement with the teacher.

### Sample training log (T = 1, α = 0.5)
```text
[T1_a0.5] step   500/3563 loss=0.3585 CE=0.4303 KL*T^2=0.2866 lr=4.57e-05 46s
[T1_a0.5] step  1000/3563 loss=0.1662 CE=0.2403 KL*T^2=0.0922 lr=3.83e-05 92s
[T1_a0.5] step  2000/3563 loss=0.1169 CE=0.1856 KL*T^2=0.0482 lr=2.33e-05 182s
[T1_a0.5] step  3000/3563 loss=0.0979 CE=0.1697 KL*T^2=0.0261 lr=8.40e-06 275s
[T1_a0.5] step  3563/3563 loss=0.0923 CE=0.1669 KL*T^2=0.0177 lr=0.00e+00 327s
[T1_a0.5] done in 327s | val acc=0.9440 val CE=0.1570 | test acc=0.9446 | test agreement w/ teacher=0.9833
```
The loss trends for all seven runs are in `results/figures/p2_loss_trends.png`.

**Table HW11.2 — the separate distillation comparison on AG News**

| Configuration | Accuracy | p50 / p95 latency (ms) | Peak memory (GB) | Size on disk (MB) |
|---|---|---|---|---|
| Teacher (BERT-base, 109.5M params) | 94.20 % | 13.87 / 21.85 | 0.929 | 418.4 |
| Best distilled student (DistilBERT, 67.0M, T=1, α=0.75) | 94.32 % | 7.82 / 11.73 | 0.771 | 256.1 |

* **Latency:** batch 1, fp32, forward only, 500 timed requests.
* **Inference time for the full test set (batch 128):** 39.1 s (194 items/s) for the teacher vs 19.5 s (390 items/s) for the student.
* **Student vs teacher:** 0.61× parameters and size, 1.77× faster p50, 2.0× throughput, 0.83× peak memory, 100 % of the teacher's accuracy.

### Per-class accuracy

| Class | Teacher | Student | Δ (pp) |
|---|---|---|---|
| World | 94.47 % | 94.26 % | **−0.21** |
| Sports | 98.68 % | 99.00 % | +0.32 |
| Business | 90.21 % | 90.58 % | +0.37 |
| Sci/Tech | 93.42 % | 93.42 % | 0.00 |

**The class that degrades most is World.** 19 items are teacher-right/student-wrong, and 15 are the reverse. The five shown are the student's most confident errors:

| Test idx | Text (abridged) | Teacher (p) | Student (p) |
|---|---|---|---|
| 528 | Passengers Stranded by Canceled Flights — British Airways cancels scores of flights at Heathrow… | World (0.58) | Business (0.91) |
| 1617 | At Last, Success on the Road for Lions — Detroit Lions end NFL road losing streak vs Bears… | World (0.88) | Sports (0.88) |
| 6698 | Terms of Endearment — Texas oilman who brought Bush into the Texas Rangers baseball club moves from ambassador to Australia to Japan… | World (0.78) | Sports (0.85) |
| 2183 | Election chaos? — Albuquerque lawyers prepare to monitor ballot counting for Kerry in New Mexico… | World (0.85) | Sci/Tech (0.84) |
| 6150 | England humble Springboks — Charlie Hodgson scores 27 points at Twickenham | World (0.98) | Sports (0.76) |

**What they have in common:** each is labelled *World*, but its dominant vocabulary belongs to another class: airlines (Business), NFL,
baseball and rugby (Sports), election law (Sci/Tech). The World label reflects the outlet or framing rather than the topic, and several items
look like label noise. The larger teacher partly fits these source-specific quirks. The 6-layer student relies more on topical keywords.

---

## Quantisation vs. distillation
GPTQ needed no training: in 47 s on a small WikiText-2 calibration set it halved OPT-125m's on-disk size (242 → 123 MB) and resident
weights (0.23 → 0.12 GB), at a cost of +4.5 perplexity. On this GPU at batch 1 it made generation about 2× slower (p50 2.10 s → 4.14 s per
128 tokens) and cut peak memory only 9 %, because a 125M-parameter model is bound by kernel launches and activations rather than by weight
bandwidth. Distillation cost a teacher logit pass plus about 5.5 min of training per student. It produced an architecturally smaller
network (6 vs 12 layers, 67M vs 110M parameters) that cut p50 latency 1.77×, doubled throughput, and shrank disk size 0.61× and
peak memory 0.83× while keeping the teacher's quality on its task. Quantisation reduces bytes per weight but keeps the same compute
graph and adds dequantisation work, while distillation removes layers, and therefore FLOPs. **For a latency-bound deployment I would try
distillation first; for a memory-bound deployment I would try GPTQ first.**
