# Real-Time Autonomous Driving Object Detection with YOLO on BDD100K

An end-to-end computer-vision pipeline for autonomous-driving perception: dataset engineering,
transfer learning, leakage-free evaluation, a TIDE-style error taxonomy, robustness analysis across
driving conditions, and latency benchmarking — in a single reproducible Colab notebook.

**Headline result:** a YOLO26-small detector fine-tuned on BDD100K reaches **mAP@50 = 0.522** and
**mAP@50–95 = 0.289** across 9 road-object classes on a held-out test split, running at
**78.6 FPS (12.7 ms)** on a Colab T4.

**Scope, stated plainly:** this is a monocular 2D object detector — one component of a perception
stack. It is not a self-driving system. No depth, no tracking, no sensor fusion, no planning or
control. See [Limitations](#limitations).

---

## Contents

- [Results](#results)
- [What the analysis found](#what-the-analysis-found)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Method](#method)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Research questions](#research-questions)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Environment and reproducibility](#environment-and-reproducibility)
- [Citation and licence](#citation-and-licence)

---

## Results

All figures below are from a single completed run and are read directly from the notebook's own
output. Nothing is estimated.

### Headline metrics — held-out test split (2,500 images)

| Metric | Value |
|---|---|
| **mAP@50** | **0.5217** |
| **mAP@50–95** | **0.2887** |
| Precision (at max-F1 confidence) | 0.6485 |
| Recall (at max-F1 confidence) | 0.4822 |
| F1 | 0.5532 |
| Mean latency | 12.72 ms |
| p95 latency | 15.36 ms |
| Throughput | 78.6 FPS |

<sub>Precision and recall are reported at the maximum-F1 confidence threshold, which is how
Ultralytics summarises them. They are not comparable to values read off the confusion matrix, which
is built at a fixed confidence. The mAP figures are threshold-independent.</sub>

### Per-class performance

| Class | AP@50 | AP@50–95 | Precision | Recall | Train instances | Share | Median box side |
|---|---|---|---|---|---|---|---|
| car | 0.752 | **0.470** | 0.757 | 0.695 | 65,859 | 54.9% | 37 px |
| bus | 0.534 | 0.408 | 0.677 | 0.471 | 1,090 | 0.9% | 75 px |
| truck | 0.553 | 0.398 | 0.586 | 0.539 | 2,807 | 2.3% | 73 px |
| traffic sign | 0.604 | 0.320 | 0.662 | 0.574 | 22,335 | 18.6% | 21 px |
| pedestrian | 0.559 | 0.266 | 0.680 | 0.498 | 8,858 | 7.4% | 33 px |
| traffic light | 0.570 | 0.208 | 0.622 | 0.583 | 17,555 | 14.7% | 16 px |
| rider | 0.377 | 0.181 | 0.633 | 0.334 | 431 | 0.4% | 40 px |
| bicycle | 0.366 | 0.176 | 0.626 | 0.296 | 640 | 0.5% | 48 px |
| motorcycle | 0.380 | **0.172** | 0.594 | 0.350 | 290 | 0.2% | 53 px |

### Robustness across driving conditions

Each partition of the held-out test split was evaluated independently.

| Attribute | Condition | Images | mAP@50 | mAP@50–95 | Precision | Recall |
|---|---|---|---|---|---|---|
| timeofday | daytime | 1,277 | 0.534 | 0.301 | 0.656 | 0.494 |
| timeofday | night | 1,034 | 0.504 | 0.273 | 0.681 | 0.447 |
| timeofday | dawn/dusk | 178 | 0.492 | 0.286 | 0.659 | 0.428 |
| weather | partly cloudy | 161 | 0.556 | 0.317 | 0.633 | 0.527 |
| weather | overcast | 306 | 0.544 | 0.291 | 0.687 | 0.484 |
| weather | rainy | 161 | 0.530 | 0.307 | 0.679 | 0.473 |
| weather | clear | 1,382 | 0.495 | 0.277 | 0.641 | 0.461 |
| weather | snowy | 190 | 0.479 | 0.266 | 0.532 | 0.470 |
| scene | city street | 1,521 | 0.534 | 0.296 | 0.668 | 0.489 |
| scene | highway | 616 | 0.472 | 0.280 | 0.696 | 0.443 |
| scene | residential | 318 | 0.475 | 0.259 | 0.595 | 0.421 |

**Night vs day: −9.3% relative mAP@50–95.** Smaller than the degradation usually assumed for
night-time detection, but read it as observational — the night partition also differs in scene mix,
class balance and object scale, so the gap bounds the combined effect rather than isolating
illumination.

### Error taxonomy (500 held-out images, IoU ≥ 0.5, conf ≥ 0.25)

6,093 true positives, 2,762 false positives, 3,157 misses — precision 0.688, recall 0.659.

| False-positive cause | Count | Share | Meaning |
|---|---|---|---|
| Localisation | 1,157 | 41.9% | object found, box not tight enough |
| Background | 970 | 35.1% | hallucination on empty scene content |
| Duplicate | 474 | 17.2% | NMS left a redundant box |
| Class confusion | 161 | 5.8% | box correct, label wrong |

**Recall by object size:**

| Size bucket | Recall | GT boxes |
|---|---|---|
| small (< 32² px) | **51.2%** | 5,052 |
| medium (32²–96²) | 79.3% | 2,998 |
| large (> 96² px) | **94.1%** | 1,200 |

Most frequent class confusions: `truck → car` (50), `car → truck` (21), `bus → truck` (16),
`traffic sign → traffic light` (16).

### Latency breakdown (Tesla T4, batch 1, fp32 eager PyTorch)

| Stage | Time |
|---|---|
| Preprocess | 1.15 ms |
| Inference | 9.96 ms |
| Postprocess (NMS) | 1.24 ms |
| **Total** | **12.72 ms → 78.6 FPS** |

End-to-end video throughput was 25.9 FPS, lower than the 51.6 FPS detection-only figure, because
decode, annotation drawing and re-encode are included there.

---

## What the analysis found

Five things the aggregate mAP number hides.

**1. Object scale dominates everything.** Recall runs 51.2% → 79.3% → 94.1% across small, medium and
large objects. Nearly **twice** the miss rate on small objects. In driving, "small" means "far away",
and far away means "matters soon" — this is the least benign way for a detector to fail.

**2. But rarity predicts per-class difficulty better than scale.** AP@50–95 correlates **+0.55** with
log training frequency and only **+0.33** with median object size. The clearest evidence is
`motorcycle`: median box side 53 px — larger than `car` at 37 px — yet the worst class in the set at
AP 0.172, on 290 training instances. Scale hurts globally; rarity hurts specific classes. They call
for different fixes, and conflating them wastes effort.

**3. The model localises worse than it detects.** Localisation is the single largest false-positive
cause at 41.9%, and the mAP@50 → mAP@50–95 gap is 0.233. The detector finds objects more reliably
than it draws tight boxes around them — consistent with running 1280×720 imagery at 640 px.

**4. Class confusions are almost entirely taxonomic, not perceptual.** `truck ↔ car ↔ bus` accounts
for most of it. These genuinely blur at distance and from behind, and the boundaries are partly
annotation conventions rather than visual facts. Some of that 5.8% is irreducible.

**5. Night costs less than expected, snow costs more.** The day→night drop is −9.3%, while the
weather spread (0.317 partly cloudy → 0.266 snowy, a 19% relative gap) is *larger* than the
time-of-day spread. Snow also shows the worst precision of any condition (0.532), suggesting
hallucinated detections on snow-textured background rather than simple misses.

---

## Quickstart

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → GPU** (a T4 is sufficient).
3. **Run all.**

No account, no token, no manual download — the notebook acquires the dataset itself (see
[Dataset](#dataset)).

**Smoke test first:** set `QUICK_TEST_MODE = True` for a ~10-minute run over tiny subsets at 3 epochs
to verify the pipeline before committing to a real run. Metrics from a quick run are meaningless and
the notebook labels them as such.

**Runtime for the results above:** ~5 min dataset acquisition + **60.4 min training** (25 epochs,
batch 24, 640 px) + ~15 min evaluation and analysis on a Colab T4.

If Colab disconnects mid-training, re-run the training cell — it resumes from `last.pt` with
optimizer state and epoch counter restored.

---

## Dataset

BDD100K availability is currently messy: the ETH Zurich mirror that the official documentation points
to (`dl.cv.ethz.ch`) **no longer resolves**. The notebook therefore tries several sources in order,
preflighting each with a DNS + HTTP `HEAD` check so a dead host costs about a second rather than
minutes of retries.

| Source | Account needed | Status |
|---|---|---|
| `manual_urls` — links you paste in | No | Works if you have portal links |
| **`hf_10k`** — `dgural/bdd100k` on Hugging Face | **No** | **Live — used for the results above** |
| `kaggle` — `solesensei/solesensei_bdd100k` | Free account | Live |
| `ethz` — official mirror | No | **Currently unreachable** |
| `huggingface` — image mirror | No | Live but no labels; opt-in only |

### What this run used

`dgural/bdd100k` — public, ungated, ~695 MB, curated by the ETH VIS Group. It ships detection boxes
**and** the `weather` / `timeofday` / `scene` attributes, so the robustness analysis works fully. It
arrives in FiftyOne format (relative `[x, y, w, h]` boxes) and the notebook converts it to the
standard BDD100K JSON + directory layout. This run converted **186,033 boxes across 10,000 images**.

### The caveat that matters

**These 10,000 images are the BDD100K *validation* split, not the 70,000-image training split.** The
notebook partitions them into its own pools:

| Pool | Images | Role |
|---|---|---|
| train | 6,428 | gradient updates |
| val | 1,071 | early stopping, best-checkpoint selection |
| **test** | **2,500** | **all reported metrics — never seen during model selection** |

The pools are disjoint (asserted in code), so **there is no leakage** and the test set genuinely
never influenced checkpoint selection. But all three come from the same official split, so this run
does not carry the extra credibility of testing against a split whose distribution the model never
saw. And with ~6.4k training images rather than 70k, absolute numbers are lower than a full-data run
would give — rare classes worst of all (`motorcycle` had just 290 instances).

To run on the full dataset, obtain BDD100K via any other route and set `DOWNLOAD_SOURCES` to skip
`hf_10k`. Nothing else changes.

### Classes

```
0 car   1 bus   2 truck   3 pedestrian   4 rider
5 bicycle   6 motorcycle   7 traffic light   8 traffic sign
```

BDD100K's tenth category, the vehicle `train`, is excluded by default: it is orders of magnitude
rarer than the others, and because mAP is a *mean over classes*, one near-empty noisy class visibly
moves the headline number without saying anything about detector quality. A stated methodological
choice, not a way to inflate the score. Set `EXCLUDED_CATEGORIES = ()` to keep it.

Both label schemas are handled — modern `det_20` (`pedestrian`, `motorcycle`, `bicycle`) and legacy
(`person`, `motor`, `bike`) — via an alias table. `drivable area` and `lane` polygons are filtered by
**geometry**, not by name, so an unexpected polygon category cannot leak into the box labels.

---

## Method

### Split design

BDD100K's official test labels are withheld for the public benchmark server, so mAP cannot be
computed on them locally. The usual workaround is to report on the val split — but if val is used for
both per-epoch checkpoint selection *and* the final number, that number is optimistically biased: the
checkpoint was chosen *because* it scored well on exactly that data.

This notebook always keeps a third pool untouched during model selection and reports only on that.
Disjointness is asserted in code rather than assumed.

Because the reported numbers come from a held-out pool rather than the benchmark server, they are
**not** directly comparable to BDD100K leaderboard entries.

### Training

Transfer learning from COCO-pretrained `yolo26s.pt` (10.01 M parameters). COCO already contains
`car`, `bus`, `truck`, `person`, `bicycle`, `motorcycle` and `traffic light`, so fine-tuning teaches
the driving-camera domain rather than the objects themselves.

| Setting | Value |
|---|---|
| Optimizer | `auto` (Ultralytics selects) |
| LR schedule | cosine, `lr0 = 0.01` |
| Mixed precision | on |
| Early stopping | patience 8 |
| Mosaic | disabled for final 5 epochs |
| Epochs | 25 configured, 25 completed, **best at epoch 20** |
| Batch | 24 (derived from measured GPU memory) |

### Error analysis

Ultralytics exposes no per-detection error breakdown, so the notebook implements its own matcher.
Predictions are sorted by confidence and greedily matched to the highest-IoU unclaimed same-class
ground truth at IoU ≥ 0.50; leftovers are categorised in the spirit of
[TIDE](https://dbolya.github.io/tide/) — duplicate, class confusion, localisation, or background.
Misses are stratified by COCO-style object size and by scene condition.

This distinction is what makes the analysis actionable: localisation errors point at resolution and
box regression, background false positives at hard negatives, class confusions at the taxonomy
itself.

### Speed benchmarking

Three things are controlled for, because naive detector timing is usually wrong:

1. **CUDA is asynchronous** — every measurement is wrapped in `torch.cuda.synchronize()`, otherwise
   you time the kernel launch, not the computation.
2. **Early iterations aren't representative** — 10 warm-up passes are discarded (cuDNN autotuning,
   memory-pool warm-up).
3. **Data loading isn't inference** — the image is decoded once and held in memory.

### Model comparison

| Model | Params | Checkpoint | Trained | mAP@50 | mAP@50–95 | Latency | FPS |
|---|---|---|---|---|---|---|---|
| yolo26s.pt | 10.01 M | 20.4 MB | yes | 0.5217 | 0.2887 | 12.72 ms | 78.6 |
| yolo26n.pt | 2.57 M | 5.5 MB | no | — | — | 12.79 ms | 78.2 |

Accuracy is reported only for models actually fine-tuned on BDD100K; the notebook prints
`not trained` rather than a plausible-looking guess. Set `RUN_MODEL_COMPARISON = True` to train the
second model under identical settings and seed.

**Worth noting from these numbers:** `yolo26n` has 3.9× fewer parameters yet measured *no faster*
(78.2 vs 78.6 FPS). At batch size 1 on a T4, the pipeline is not backbone-bound — 2.4 ms of the
12.7 ms is fixed pre/post-processing, and the small model cannot amortise the Python and NMS
overhead. A capacity comparison at this batch size measures the harness as much as the architecture;
batched or exported (TensorRT) inference would separate them properly.

---

## Configuration

Everything tunable lives in **one cell**. Nothing else needs editing.

```python
MODEL_NAME              = "yolo26s.pt"   # falls back to yolo11s / yolov8s on older Ultralytics
COMPARISON_MODEL_NAME   = "yolo26n.pt"
RUN_MODEL_COMPARISON    = False

IMAGE_SIZE              = 640
BATCH_SIZE              = 0              # 0 => derive from measured GPU memory
EPOCHS                  = 25
LEARNING_RATE           = 0.01
PATIENCE                = 8
AMP                     = True

CONFIDENCE_THRESHOLD    = 0.25           # deployment-style threshold
IOU_THRESHOLD           = 0.7            # NMS
EVAL_IOU_MATCH          = 0.50           # error-analysis matcher

TRAIN_IMAGES            = 12000          # None => use the whole available split
VAL_IMAGES              = 2000
TEST_IMAGES             = 5000

DOWNLOAD_SOURCES        = ("manual_urls", "hf_10k", "kaggle", "ethz", "huggingface")
HF_10K_REPO             = "dgural/bdd100k"
MANUAL_DOWNLOAD_URLS    = ()
HALT_IF_DATASET_MISSING = True           # stop loudly rather than skip cells quietly

SEED                    = 42
USER_IMAGE_PATH         = None           # your own dashcam frame
USER_VIDEO_PATH         = None           # your own driving video
```

`IMAGE_SIZE = 640` halves the linear resolution of 1280×720 source imagery, which the error analysis
shows costs most on small objects. `960` measurably improves small-object recall at roughly 2× the
training cost — the single highest-value change available.

---

## Outputs

The pipeline is **idempotent**: re-running reuses the cached annotation index, skips the dataset build
when the configuration signature is unchanged, and skips training when finished weights exist
(`FORCE_RETRAIN = True` overrides). Changing a split size clears and rebuilds cleanly rather than
merging with the previous build.

| Location | Contents |
|---|---|
| `models/` | `<run>_best.pt`, `<run>_last.pt` |
| `results/` | test metrics, per-class CSV, condition CSV, error-analysis JSON, per-image errors, speed benchmark, model comparison, experiment summary, confusion matrix |
| `visualizations/` | dataset statistics, ground-truth samples, training curves, per-class analysis, confusion matrix, error analysis, failure cases, condition analysis |
| `predictions/` | random and difficult-scene prediction grids, single-image inference, annotated MP4 |
| `logs/` | config snapshot, `results.csv`, `args.yaml` |
| `cache/` | gzipped annotation index, dataset build manifest |

Training runs on local Colab disk for speed (Drive is a network filesystem and slow for the many
small reads training generates), and durable artifacts are mirrored to Drive as each stage completes.

---

## Research questions

The notebook answers each from measured results.

**RQ1 — Accuracy.** mAP@50 0.522 / mAP@50–95 0.289 across 9 classes on 2,500 held-out images. The
0.233 gap between them says the model finds objects more reliably than it localises them tightly.

**RQ2 — Class difficulty.** `car` best (AP 0.470), `motorcycle` worst (AP 0.172). Frequency correlates
+0.55 with AP, size +0.33 — rarity is the stronger predictor, pointing at class weighting or
oversampling before resolution changes. With 9 classes these correlations are indicative, not
conclusive, and the two factors are confounded.

**RQ3 — Robustness.** Night costs −9.3% relative to day; the weather spread (0.266 snowy → 0.317
partly cloudy) is wider than the time-of-day spread. Observational, not causal.

**RQ4 — Accuracy/latency.** 12.7 ms mean, 15.4 ms p95, 78.6 FPS for mAP@50–95 0.289 on a T4.

**RQ5 — Failure modes.** Localisation dominates false positives (41.9%); small-object recall is
51.2% against 94.1% for large. mAP treats every miss identically, but driving does not — a missed
distant traffic sign and a missed pedestrian at close range are both "one false negative" to the
metric and are not remotely equivalent on the road. A deployed system would be evaluated on
distance-stratified, safety-weighted metrics.

---

## Limitations

**Reduced training data.** ~6.4k training images rather than 70k, because the acquisition route that
works without an account provides 10,000 images. Absolute numbers are lower than a full-data run and
rare classes suffer most.

**Train and test share an official split.** Disjoint and leakage-free, but drawn from the same
BDD100K val split — so the evaluation lacks the extra credibility of an unseen distribution.

**Not comparable to the leaderboard.** Official test labels are withheld.

**Geographic bias.** BDD100K was collected in New York and the SF Bay Area. Vehicle fleets, signage,
road furniture and pedestrian behaviour differ elsewhere; nothing here predicts that degradation.

**Imperfect annotations.** Human-annotated at scale, with missing boxes, inconsistent occlusion
handling and ambiguous class assignments. Some measured false positives are correct detections of
unlabelled objects — a ceiling on measurable precision no model improvement can exceed.

**Condition comparisons are observational.** Partitions are not matched on anything but the attribute
itself.

**Small-object detection is weak** (51.2% recall), and in driving, small means far away.

**Occlusion and crowding degrade recall.** A single-frame detector cannot reason about an object it
can only partly see; temporal tracking is the standard remedy and is absent here.

**Single seed, single run.** No confidence intervals; run-to-run variance in detection training is
not negligible.

**Detection is not driving.** This system outputs 2D boxes on single frames. It has no depth or 3D
understanding (no idea whether a car is 5 m or 50 m away — the most decision-relevant quantity in the
scene), no temporal reasoning, no traffic-light *state* (it detects that a light exists, not whether
it is red), no prediction, no planning or control, and no safety case: no redundancy, no failure
detection, no operational design domain, no validation against ISO 26262 or ISO 21448/SOTIF.

**This is a perception component and an evaluation methodology, not a self-driving system, and its
outputs should not be used to control a vehicle.**

---

## Future work

Ordered by effort-to-value, informed by the failure modes actually measured.

**Addressing what the analysis found**
- **Higher input resolution** (960 or 1280) — targets the 51.2% small-object recall directly; the
  highest-value single change.
- **Class-imbalance handling** — `cls_pw` inverse-frequency weighting or oversampling images
  containing `rider`, `bicycle`, `motorcycle`, justified by the +0.55 frequency correlation.
- **Full 70k training set** with a longer schedule.
- **Tiled inference** on the upper-central image region where distant small objects concentrate.

**Extending the stack**
- **Multi-object tracking** (ByteTrack/BoT-SORT, built into Ultralytics via `model.track()`) —
  recovers objects missed in individual frames and supplies velocity. Highest-value architectural
  addition.
- Monocular depth estimation, to lift 2D boxes toward metric distance.
- Lane and drivable-area segmentation — BDD100K ships both annotation types already.
- 3D detection, stereo vision, BEV perception.
- Sensor fusion with radar and lidar.

**Deployment**
- TensorRT/ONNX export with INT8 quantisation, benchmarked on embedded hardware under sustained
  thermal load — and at batch > 1, which would properly separate the `n`/`s` capacity comparison.
- Distillation from a larger teacher.
- Domain-shift evaluation on Cityscapes, nuScenes or KITTI.
- Multi-seed runs with confidence intervals.

---

## Environment and reproducibility

| Component | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.147 |
| GPU | Tesla T4 (14.6 GiB, sm_75) |

`seed = 42`, `deterministic = True`. The exact configuration for every run is written to
`logs/config_<run_name>.json` alongside `results/experiment_summary_<run_name>.json`. Point a fresh
Colab session at the same Drive folder to reproduce, or delete `models/` to retrain from scratch.

---

## Citation and licence

```bibtex
@inproceedings{bdd100k,
  title     = {BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning},
  author    = {Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying
               and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

**BDD100K** is released for educational, research and not-for-profit use under the
[BDD100K licence](https://doc.bdd100k.com/license.html); you agree to it by downloading. Copyright
© 2018 The Regents of the University of California. Contact UC Berkeley's Office of Technology
Licensing for commercial use. This repository redistributes no data.

**Ultralytics YOLO** is AGPL-3.0 licensed, with commercial licences available separately — worth
checking before building anything commercial on this.

The notebook in this repository is released under the MIT licence.
