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

```
xd2-net/
├── xd2net/                    ← installable package
│   ├── config.py              ← dataclasses + YAML loader
│   ├── data/
│   │   ├── preprocessing.py   ← data loading, encoding, vectorization
│   │   ├── feature_engineering.py  ← build input_exp + feature names
│   │   └── dataset.py         ← torch Dataset + DataLoader helpers
│   ├── model/
│   │   ├── encoder.py         ← InputProjection, EventEncoder,
│   │   │                          TemporalAttentionPool, FeatureWeightingBlock
│   │   ├── heads.py           ← ThetaHead, PredictionHead
│   │   └── xd2net.py          ← top-level XD2Net nn.Module
│   ├── training/
│   │   ├── trainer.py         ← training loop, early stopping, LR scheduler
│   │   └── metrics.py         ← AUC, F1, feature agreement
│   └── explain/
│       └── explainer.py       ← extract thetas + alpha → DataFrames
├── configs/
│   └── bpic2012.yaml          ← BPIC 2012 default config
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_train.ipynb
│   ├── 03_evaluate.ipynb
│   └── 04_explain.ipynb
├── scripts/
│   └── run_experiment.py      ← headless CLI
├── data/
│   └── BPIC_2012_OUTCOME.csv  ← place dataset here
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

| Notebook | Purpose | Output |
|---|---|---|
| `01_data_preparation.ipynb` | Load CSV, split, encode, vectorize, save per-bucket artifacts | `outputs/artifacts/bucket_1_*.pkl` |
| `02_train.ipynb` | Train one XD2Net per bucket, save checkpoints | `outputs/checkpoints/model_1_*.pt` |
| `03_evaluate.ipynb` | Test metrics for XD2-Net + LR baseline | `outputs/results/comparison_results.csv` |
| `04_explain.ipynb` | Extract thetas, alpha, feature agreement, effective complexity | `outputs/explanations/` |

### 4. GPU acceleration (optional)

Colab's free T4 GPU works automatically. The config has `device: "auto"` which selects CUDA when available. No changes needed.

---

## Configuration

Edit `configs/bpic2012.yaml` to change dataset paths, model size, or training schedule:

```yaml
data:
  csv_path: "data/BPIC_2012_OUTCOME.csv"
  cat_dynamic: ["ACTIVITY", "ROLE"]
  num_dynamic: ["TASK_DURATION", "TIME_ELAPSED"]
  num_static:  ["AMOUNT_REQ"]
  prefix_lengths: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

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
  device: "auto"      # auto | cpu | cuda | mps
```

To use a different event log, create a new YAML file and point `csv_path` at your CSV. The expected columns are controlled by the `caseid_col`, `activity_col`, and `outcome_col` fields. Your CSV needs a timestamp column for event ordering and a resource column (`RESOURCE`) for role discovery.

---

## Headless CLI (non-Colab)

```bash
pip install -e .
python scripts/run_experiment.py --config configs/bpic2012.yaml

# Train only specific buckets
python scripts/run_experiment.py --config configs/bpic2012.yaml --buckets 1_10 1_15
```

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

### Level-1: TemporalAttentionPool

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
