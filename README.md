# XD²-Net (Transformer)

Inherently explainable process outcome prediction, refactored from TensorFlow/Keras + Colab scratchpad into a proper Python package with a PyTorch Transformer backbone.

## What changed

| | Original | Refactored |
|---|---|---|
| Framework | TensorFlow 2 / Keras | PyTorch 2.1+ |
| Sequence backbone | Alpha + Beta BiLSTM (70 units) | Transformer encoder (d_model=140) |
| Level-1 explainability | alpha/beta LSTM attention weights | TemporalAttentionPool + FeatureWeightingBlock |
| Level-2 explainability | softmax thetas layer | Same — preserved exactly |
| Prediction formula | `sigmoid(W · (thetas ⊙ input_exp))` | Same — W initialized to ones |
| Code structure | Colab notebooks + loose scripts | Installable `xd2net` package |
| Paths | Hardcoded Drive paths | Config YAML + env detection cell |
| Memory bug | `np.concatenate` in loop (O(N²)) | Pre-allocated arrays |
| Data leakage | Normalizer refitted on test set | Fit on train, apply to test |

The explainability contract is unchanged: thetas are always non-negative, sum to 1 (softmax), and each theta_i directly scales feature_i in the final prediction, so causal attribution is exact — not post-hoc.

---

## Directory layout

Two model variants sit side by side and share everything except the encoder that produces the
Level-1 explanation:

* **`baseline`**: Transformer encoder + external `TemporalAttentionPool` (`XD2Net`).
* **`hierarchical`**: hierarchical self-attention, event → field and case → event (`XD2NetHier`).
  See `docs/hierarchical_attention/`.

Both packages have the same module set (`config`, `data/features`, `layers`, `model`, `training`,
`explainer`, `spec`). Shared code lives in `xd2net/common`.

```
xd2-net/
├── xd2net/                       ← installable package
│   ├── common/                   ← shared by every variant
│   │   ├── config.py             ← TrainingConfig, DatasetConfig, ExperimentConfig, YAML loaders
│   │   ├── data/
│   │   │   ├── eventlog.py       ← load any event log (columns, order, outcome, time features)
│   │   │   ├── preprocessing.py  ← split, role discovery, encoding, prefixes, vectorization
│   │   │   ├── features.py       ← EventSchema, static input, explainable (exp) input
│   │   │   ├── pipeline.py       ← prepare_dataset(), build_bucket(prepared, n, encoder)
│   │   │   └── dataset.py        ← torch Dataset + DataLoader helpers
│   │   ├── heads.py              ← ThetaHead, PredictionHead (Level-2)
│   │   ├── masks.py              ← padding mask
│   │   ├── metrics.py            ← AUC, F1, feature agreement
│   │   ├── training.py           ← Trainer: early stopping, LR schedule, best checkpoint
│   │   ├── explainer.py          ← Level-2 thetas + Level-1 attention-cell tables
│   │   └── registry.py           ← ModelSpec, get_spec("baseline" | "hierarchical")
│   ├── baseline/                 ← XD2Net
│   │   ├── config.py  data/features.py (one-hot)  layers.py  model.py
│   │   └── training.py  explainer.py  spec.py
│   └── hierarchical/             ← XD2NetHier
│       ├── config.py  data/features.py (field tokens)  layers.py  tokenizer.py  model.py
│       └── training.py  explainer.py  spec.py
├── configs/
│   ├── models/{baseline,hierarchical}.yaml   ← model + training settings per variant
│   └── datasets/{bpic2012,_template}.yaml    ← one per event log, shared by both models
├── notebooks/                    ← 01–04 take MODEL / DATASET_CONFIG; 05 = hierarchical attention
├── scripts/
│   └── run_experiment.py         ← headless CLI (one or more models × datasets)
├── tests/                        ← pytest: common/, baseline/, hierarchical/, test_registry.py
├── docs/
├── data/
│   └── BPIC_2012_OUTCOME.csv     ← place dataset here
├── setup.cfg
├── pyproject.toml
└── requirements.txt
```

---

## Running on Google Colab

### 1. Upload the repo to Google Drive

Upload the entire `xd2-net/` folder to your Drive, e.g. at:
```
My Drive/xd2-net/
```

### 2. Upload the dataset

Place `BPIC_2012_OUTCOME.csv` at:
```
My Drive/xd2-net/data/BPIC_2012_OUTCOME.csv
```

### 3. Open notebooks in order

Each notebook starts with a detection cell that mounts Drive and installs the package automatically:

```python
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    REPO_ROOT = '/content/drive/MyDrive/xd2-net'
    %pip install -e {REPO_ROOT} --quiet
else:
    REPO_ROOT = os.path.abspath('..')
    sys.path.insert(0, REPO_ROOT)
```

Run notebooks in this order:

Set `MODEL` (`baseline` or `hierarchical`) and `DATASET_CONFIG` in the parameter cell of each
notebook, then run them in this order. Paths below are under `outputs/<MODEL>/<dataset>/`.

| Notebook | Purpose | Output |
|---|---|---|
| `01_data_preparation.ipynb` | Load the log, split, encode, vectorize, save per-bucket artifacts | `artifacts/bucket_1_*.pkl` |
| `02_train.ipynb` | Train one model per bucket, save the best checkpoint + config | `checkpoints/model_1_*.pt`, `model_config_1_*.json` |
| `03_evaluate.ipynb` | Test metrics for XD2-Net + LR baseline | `results/comparison_results.csv` |
| `04_explain.ipynb` | Extract thetas, alpha, feature agreement, effective complexity | `explanations/` |
| `05_hier_explain.ipynb` | Hierarchical attention (α, β, α·β), reading `run_experiment.py` outputs | plots |

### 4. GPU acceleration (optional)

Colab's free T4 GPU works automatically. The config has `device: "auto"` which selects CUDA when available. No changes needed.

---

## Configuration

Two kinds of YAML file are used:

* **Model config:** `configs/models/<model>.yaml` holds the `model` and `training` settings for one
  variant. It is applied unchanged to every dataset.
* **Dataset config:** `configs/datasets/<name>.yaml` holds one event log's schema, outcome and
  prefix settings. It is shared by both models. Copy `configs/datasets/_template.yaml` (fully
  annotated) to add a log.

```yaml
# configs/datasets/bpic2012.yaml (excerpt)
name: bpic2012
path: data/BPIC_2012_OUTCOME.csv
columns: {caseid: CASEID, timestamp: END_TIMESTAMP, activity: MILESTONE, role: null, resource: RESOURCE}
event_order: file                 # keep file order within each case
event_num: [TASK_DURATION, TIME_ELAPSED]
case_num: [AMOUNT_REQ]
outcome: {mode: column, column: OUTCOME, positive: A_APPROVED, labels: [A_DECLINED, A_APPROVED]}
prefix: {lengths: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
```

```yaml
# configs/models/baseline.yaml (excerpt)
model:
  d_model: 140        # 2×70 — matches original BiLSTM hidden size
  nhead: 4
  num_encoder_layers: 1
  dropout: 0.15
training:
  learning_rate: 0.01
  batch_size: 256
  epochs: 100
  early_stopping_patience: 10
  l2_reg: 0.0001
  seed: 42
  device: "auto"      # auto | cpu | cuda | mps
```

Any event log works if it has a case id, a timestamp (or a meaningful file order) and an
activity, plus either a role column or a resource column for role discovery. Each extra variable
is declared as case-level or event-level, categorical or numeric. The outcome is either a column
or "the case contains one of these activities"; in the activity mode, prefixes are cut before the
first target activity so the label can't leak.

---

## Headless CLI (non-Colab)

```bash
pip install -e .

# Both models on BPIC 2012, all buckets
python scripts/run_experiment.py --model baseline hierarchical --datasets configs/datasets/bpic2012.yaml

# One model, several logs, selected buckets
python scripts/run_experiment.py --model hierarchical \
    --datasets configs/datasets/bpic2012.yaml configs/datasets/my_log.yaml --buckets 1_10 1_15
```

Each dataset is prepared once and shared by every model, so they see the same split and
prefixes. The outputs go to `outputs/<model>/<dataset>/`, and the combined
`outputs/results_all.csv` has one row per (model, dataset, bucket). Other options:
`--model-config` (one per model), `--seed`, `--epochs`, `--workspace` and `--fail-fast`.

Run the tests with `pytest tests -q`.

---

## Architecture notes

### Transformer mapping to original BiLSTMs

```
Original:
  raw_input → Alpha BiLSTM(70) → Dense(1) → Softmax(time) → alpha (B,T,1)
  raw_input → Beta  BiLSTM(70) → Dense(D,tanh) → Softmax(time) → beta (B,T,D)
  c_t = sum(alpha × beta × raw_input, axis=T)

New:
  raw_input → InputProjection → TransformerEncoder (d_model=140, 1 layer, 4 heads)
  enc_out → TemporalAttentionPool  → alpha_weights (B,T)    [Level-1]
  enc_out → FeatureWeightingBlock  → beta_weights  (B,T,D)  [Level-1]
  c_t = sum(alpha.unsqueeze(-1) × beta × raw_input, axis=T)   ← identical formula
```

The context vector assembly formula is byte-for-byte identical to the original, preserving the mathematical meaning of both explainability levels.

This is the `baseline` variant. The `hierarchical` variant replaces the encoder and both pooling
blocks with two levels of self-attention (field tokens → `[EVT]`, event representations → `[CASE]`)
and reads α and β from their attention rows. See `docs/hierarchical_attention/APPROACH.md`.

### Level-1 (baseline): TemporalAttentionPool

Single-head cross-attention with one learned query vector attending over the encoder output. Produces `alpha_weights (B, T)` — softmax over non-padded positions. Directly analogous to the original `Softmax(axis=1)` on the alpha BiLSTM output.

Using a dedicated pooling attention (rather than the encoder's internal self-attention weights) is intentional: self-attention produces a `(T,T)` matrix, which requires aggregation to get per-token salience. The dedicated pooling head gives a clean scalar per timestep directly, matching the original semantics.

### Level-2: ThetaHead (unchanged)

```python
thetas = softmax(Dense(exp_dim)(softmax(Dense(1.5×exp_dim)(c_full))))
prediction = sigmoid(W · (thetas ⊙ input_exp))   # W initialized to ones
```

Thetas are softmax-constrained: non-negative, sum to 1. Each theta_i represents the exact fractional contribution of aggregated feature_i to the final prediction — no approximation.

---

## Explanation outputs

**`df_predictions`** (per case):
```
CASEID | PREDICTION_PROB | PREDICTION | GROUND_TRUTH | BUCKET
```

**`df_explanations`** (per case × feature, long format):
```
CASEID | FEATURE | FEATURE_VALUE | FEATURE_IMPORTANCE | BUCKET
```
`FEATURE_IMPORTANCE = theta_i × final_layer_weight_i`

**`alpha_weights`** (N × T numpy array): event-level attention for temporal heatmaps.

**`attention_cells`** (per case × real event × field, long format), the same schema for both models:
```
CASEID | EVENT_POS | EVENT_INDEX | EVENT_ACTIVITY | EVENT_ROLE | FIELD | VALUE_LABEL | VALUE_NUM |
EVENT_ATTN | FEATURE_ATTN | JOINT_ATTN | EVT_SELF_ATTN | BUCKET
```
`EVENT_POS` 0 is the most recent event; `EVENT_INDEX` 0 is the first. `EVENT_ATTN` is α. The
baseline has no per-field attention, so its `FEATURE_ATTN` / `JOINT_ATTN` / `EVT_SELF_ATTN` are NaN.
The hierarchical model fills them with β, α·β and the `[EVT]` self-mass.

---

## Requirements

```
torch>=2.1.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
pyyaml>=6.0
matplotlib>=3.7
seaborn>=0.13
plotly>=5.17
torchinfo>=1.8
```
