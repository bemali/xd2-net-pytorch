# Hierarchical Self-Attention XD2-Net — Development Spec

Status: ready for implementation · Branch: `feattransformer-self-attention-exp`
Rationale and math: [`APPROACH.md`](APPROACH.md)

---

## 1. Scope

**In scope**
* New subpackage `xd2net/hierarchical/` that implements the two-level self-attention encoder,
  value-aware tokenization, attention extraction, and hierarchical explanation outputs.
* New config dataclass, experiment script, notebook, and tests for the variant.

**Out of scope / constraints**
* **No modification** of baseline modules: `xd2net/model/*`, `xd2net/explain/explainer.py`,
  `xd2net/training/*`, `xd2net/data/*`, `xd2net/config.py`. They may be *imported* and
  subclassed, not edited.
* The Level-2 explainer refactor in `xd2net/explainer_edits.md` is separate work. This spec must
  stay compatible with it (it does, because the heads are unchanged).

---

## 2. File layout

```
xd2net/hierarchical/
├── __init__.py          exports XD2NetHier, HierModelConfig, TokenSchema, HierExplainer
├── config.py            HierModelConfig, HierConfig, load_hier_config()
├── data.py              TokenSchema, build_token_schema(), build_dynamic_tokens()
├── layers.py            AttnEncoderLayer, AttnEncoder, cls_readout(), attention_rollout()
├── tokenizer.py         FieldTokenizer
├── model.py             EventLevelEncoder, CaseLevelEncoder, HierOutput, XD2NetHier
└── explainer.py         HierExplainer(Explainer)
configs/bpic2012_hier.yaml
scripts/run_hier_experiment.py
notebooks/05_hier_explain.ipynb
tests/hierarchical/
├── test_data.py
├── test_layers.py
├── test_model.py
└── test_explainer.py
```

Reused from the baseline without changes: `ThetaHead`, `PredictionHead`
(`xd2net/model/heads.py`), `make_padding_mask` (`xd2net/model/encoder.py`), `EventLogDataset`,
`make_dataloaders`, `make_test_loader` (`xd2net/data/dataset.py`), `Trainer`
(`xd2net/training/trainer.py`), `Explainer` (`xd2net/explain/explainer.py`), and all of
`xd2net/data/preprocessing.py` plus the `input_exp` / static builders in `feature_engineering.py`.

---

## 3. Configuration — `xd2net/hierarchical/config.py`

```python
@dataclass
class HierModelConfig:
    d_model: int = 64
    nhead: int = 4                    # must divide d_model
    event_layers: int = 1             # Level A depth
    case_layers: int = 1              # Level B depth
    dim_feedforward_mult: int = 4
    dropout: float = 0.15
    attn_readout: str = "rollout"     # "rollout" | "last"  (identical when layers == 1)
    dense_1_mult: float = 1.5         # ThetaHead
    l2_reg: float = 0.0001            # applied by Trainer to model.encoder params

@dataclass
class HierConfig:
    data: DataConfig                  # reused from xd2net.config
    model: HierModelConfig
    training: TrainingConfig          # reused
    output: OutputConfig              # reused

def load_hier_config(path: str) -> HierConfig: ...
```

* `load_hier_config` reuses `xd2net.config._from_dict`.
* Validation in `__post_init__`: `d_model % nhead == 0`, `attn_readout in {"rollout","last"}`,
  layers ≥ 1.
* `configs/bpic2012_hier.yaml` = `configs/bpic2012.yaml` with the `model:` block replaced by the
  fields above. Set `output.workspace_dir: ./outputs_hier` so results never overwrite baseline
  outputs.

> Note: `d_model` defaults to 64 rather than 140 because Level A only sees 5 tokens. Treat it as a
> tunable hyperparameter.

---

## 4. Data — `xd2net/hierarchical/data.py`

### 4.1 `TokenSchema`

```python
@dataclass(frozen=True)
class TokenSchema:
    cat_fields: List[str]                    # e.g. ["ACTIVITY", "ROLE"]
    num_fields: List[str]                    # e.g. ["TASK_DURATION", "TIME_ELAPSED"]
    cardinalities: List[int]                 # per cat field, EXCLUDING pad (embedding size = card+1)
    index_to_label: Dict[str, Dict[int, str]]  # per cat field, shifted index (≥1) → label

    @property
    def n_cat(self) -> int
    @property
    def n_num(self) -> int
    @property
    def field_names(self) -> List[str]       # cat_fields + num_fields  (packed column order)
    @property
    def n_fields(self) -> int

    def to_dict(self) -> dict / @classmethod from_dict(d) -> TokenSchema   # JSON-serialisable
```

### 4.2 Builders

```python
def build_token_schema(indexes: Dict, cat_dynamic: List[str], num_dynamic: List[str]) -> TokenSchema
```
* `cardinalities[i] = len(indexes["dynamic_indexes"][f"{feat}_index"])`.
* `index_to_label[feat] = {orig_idx + 1: label for label, orig_idx in index.items()}`.

```python
def build_dynamic_tokens(vec: dict, schema: TokenSchema, max_len: int, lengths: np.ndarray) -> np.ndarray
```
* Returns float32 `(N, max_len, n_cat + n_num)`.
* Columns `[0, n_cat)`: `vec["prefixes"][feat][:, :max_len] + 1`, then **set to 0 at every
  position `t ≥ lengths[i]`** (PAD). Use `lengths`, not the value 0, because 0 is a valid original
  index.
* Columns `[n_cat, n_cat+n_num)`: `vec["prefixes"][feat][:, :max_len]` (already normalised; zero
  at padding).
* The result goes into `inputs["input_dynamic"]` in place of the one-hot tensor. Everything else
  in the bucket dict (`input_static`, `input_exp`, `lengths`, `caseid`, `feature_names`) is built
  with the existing functions.

### 4.3 Invariants (asserted in the builder)
* Category columns hold integers in `[0, card]`; value 0 appears **only** at padded positions.
* `max(index) < 2**24` (exact float32 representation).

---

## 5. Layers — `xd2net/hierarchical/layers.py`

### 5.1 `AttnEncoderLayer`
A pre-norm Transformer encoder layer that **returns attention weights**.
`nn.TransformerEncoderLayer` calls MHA with `need_weights=False`, so it cannot be used.

```python
class AttnEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout): ...
        # norm1, self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True),
        # norm2, ff = Linear→GELU→Dropout→Linear, dropout1, dropout2
    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """x: (N, L, d) → (out (N, L, d), attn (N, H, L, L))"""
        h = self.norm1(x)
        a, w = self.self_attn(h, h, h, key_padding_mask=key_padding_mask,
                              need_weights=True, average_attn_weights=False)
        x = x + self.dropout1(a)
        x = x + self.dropout2(self.ff(self.norm2(x)))
        return x, w
```
* Attention dropout applies to `w` before it multiplies V. The **returned** `w` is the
  post-softmax, pre-dropout distribution in eval mode (PyTorch returns pre-dropout weights).
  Explanations are always extracted in `eval()` mode.

### 5.2 `AttnEncoder`
```python
class AttnEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout): ...
        # layers: ModuleList[AttnEncoderLayer], final_norm: LayerNorm
    def forward(self, x, key_padding_mask=None) -> Tuple[Tensor, List[Tensor]]:
        """returns (final_norm(x), [attn_l  (N,H,L,L) for each layer])"""
```

### 5.3 Readout helpers
```python
def attention_rollout(attns: List[Tensor]) -> Tensor:
    """attns: list of (N,H,L,L). Head-mean, add identity: Â = 0.5·mean_h(A) + 0.5·I,
    row-normalise, R = Â_L @ … @ Â_1. Returns R (N,L,L)."""

def cls_readout(attns: List[Tensor], mode: str,
                token_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Return (weights (N, L-1), self_mass (N,)).
    mode="last":    row = mean_h(attns[-1])[:, 0, :]
    mode="rollout": row = attention_rollout(attns)[:, 0, :]
    self_mass = row[:, 0]; w = row[:, 1:]
    if token_mask is not None (True = pad): w = w.masked_fill(token_mask, 0)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)"""
```

---

## 6. Tokenizer — `xd2net/hierarchical/tokenizer.py`

```python
class FieldTokenizer(nn.Module):
    def __init__(self, schema: TokenSchema, d_model: int): ...
        # field_emb:   nn.Embedding(n_fields, d_model)
        # val_emb:     nn.ModuleList([nn.Embedding(card+1, d_model, padding_idx=0) for card in cardinalities])
        # num_w, num_b: nn.Parameter(n_num, d_model) each  (init normal std=0.02 / zeros)
    def forward(self, packed: Tensor) -> Tensor:
        """packed (B, T, n_cat+n_num) float → tokens (B, T, n_fields, d_model)
        cat:  val_emb[i](packed[..., i].long())
        num:  packed[..., n_cat+j, None] * num_w[j] + num_b[j]
        tokens = stack([...], dim=2) + field_emb.weight  (broadcast over B,T)"""
```

---

## 7. Model — `xd2net/hierarchical/model.py`

### 7.1 Level encoders
```python
class EventLevelEncoder(nn.Module):        # Level A
    # evt_token: Parameter(1,1,d); encoder: AttnEncoder(event_layers)
    def forward(self, tokens: Tensor, readout: str) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """tokens (B,T,F,d) → reshape (B·T,F,d) → prepend [EVT] → (B·T,1+F,d)
        → AttnEncoder (no mask) → evt_rep = out[:,0]  reshape (B,T,d)
        beta, evt_self = cls_readout(attns, readout)   reshape (B,T,F), (B,T)
        returns (evt_rep, beta, evt_self, attns)"""

class CaseLevelEncoder(nn.Module):         # Level B
    # case_token: Parameter(1,1,d); pos_emb: nn.Embedding(max_len, d); encoder: AttnEncoder(case_layers)
    def forward(self, evt_rep, pad_mask, readout) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """x = evt_rep + pos_emb(arange(T))   (pos zeroed at pad, as baseline InputProjection)
        prepend [CASE] → (B,1+T,d); kpm = cat([False], pad_mask) (B,1+T)
        → AttnEncoder(kpm) → case_rep = out[:,0] (B,d)
        alpha, case_self = cls_readout(attns, readout, token_mask=pad_mask)
        returns (case_rep, alpha, case_self, attns)"""
```

### 7.2 Output container
```python
class HierOutput(NamedTuple):
    thetas: Tensor           # (B, exp_dim)
    act_output: Tensor       # (B,)   probabilities
    alpha: Tensor            # (B, T)    case → event, 0 at pad, sums to 1
    beta: Tensor             # (B, T, F) event → field, 0 at pad events, sums to 1 per real event
    joint: Tensor            # (B, T, F) alpha[...,None] * beta, sums to 1 per case
    evt_self_attn: Tensor    # (B, T)   [EVT] self-mass (diagnostic), 0 at pad
    case_self_attn: Tensor   # (B,)     [CASE] self-mass (diagnostic)
    event_attn_raw: List[Tensor]  # per layer (B·T, H, 1+F, 1+F)
    case_attn_raw: List[Tensor]   # per layer (B, H, 1+T, 1+T)
```

### 7.3 `XD2NetHier`
```python
class XD2NetHier(nn.Module):
    def __init__(self, schema: TokenSchema, dim_static: int, exp_dim: int, max_len: int,
                 cfg: Optional[HierModelConfig] = None): ...
        # self.encoder = nn.ModuleDict({"tokenizer": FieldTokenizer, "event": EventLevelEncoder,
        #                               "case": CaseLevelEncoder})
        # self.theta_head = ThetaHead(d_model + dim_static, exp_dim, cfg.dense_1_mult)
        # self.prediction_head = PredictionHead(exp_dim)

    def forward(self, raw_input, raw_input_static, input_exp, lengths=None, return_attn=False):
        """raw_input: packed (B,T,n_cat+n_num). Same positional signature as XD2Net.
        return_attn=False → (thetas, act_output, alpha)    # Trainer/Explainer compatible
        return_attn=True  → HierOutput"""
```
* `self.encoder` **must** hold every encoder parameter, because `Trainer._train_epoch` applies
  L2 to `self.model.encoder.parameters()` (`xd2net/training/trainer.py:163`).
* `self.prediction_head.linear` must keep this exact attribute path, because `Explainer`
  reads `model.prediction_head.linear.weight`.
* If `lengths is None`, treat every position as real (`pad_mask = None`).
* Store `schema` and `cfg` on the module. Add `get_config() -> dict` (schema dict +
  cfg asdict + dims) so that a checkpoint can be rebuilt; the script saves it as JSON next to the
  checkpoints.

---

## 8. Explainer — `xd2net/hierarchical/explainer.py`

```python
class HierExplainer(Explainer):
    # inherits explain_loader(), global_explanation() unchanged (Level-2)
    def get_temporal_attention(...) -> np.ndarray          # inherited; returns alpha (N,T)

    def get_hierarchical_attention(self, raw_input, raw_input_static, input_exp, lengths)
        -> Dict[str, np.ndarray]   # keys: alpha, beta, joint, evt_self_attn, case_self_attn

    def explain_attention_loader(self, loader, caseids, schema: TokenSchema, bucket: str,
                                 raw_num_values: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Returns (df_cells, df_case_fields)."""

    def global_attention(self, df_cells, by: str = "field") -> pd.DataFrame   # by ∈ {"field", "field_label"}
```

**`df_cells`** — one row per (case, real event, field):

| Column | Description |
|---|---|
| CASEID | case id |
| EVENT_POS | 0-based position in prefix |
| FIELD | field name |
| VALUE_LABEL | label for categorical fields (via `schema.index_to_label`); empty for numeric |
| VALUE_NUM | model-space value for numeric fields; NaN for categorical |
| VALUE_RAW | raw numeric value, if `raw_num_values` is given |
| EVENT_ATTN | α[t] |
| FEATURE_ATTN | β[t,f] |
| JOINT_ATTN | α[t]·β[t,f] |
| EVT_SELF_ATTN | [EVT] self-mass for event t |
| BUCKET | bucket key |

Padded events are **excluded** (not written as zero rows).

**`df_case_fields`** — one row per (case, field): `CASEID, FIELD, CASE_FIELD_ATTN (Σ_t JOINT_ATTN),
CASE_SELF_ATTN, BUCKET`.

**`global_attention`** — mean of `CASE_FIELD_ATTN` per field (`by="field"`), or the mean over
cases of Σ_{t: label=v} JOINT_ATTN per (field, label) (`by="field_label"`). Also report the
number of cases containing each label. Sorted descending.

Validation (raise `ValueError`): `len(caseids)` equals the number of rows produced by the loader.
The loader must not shuffle (check `isinstance(loader.sampler, SequentialSampler)`, otherwise
warn). The schema field count must equal `raw_input.shape[-1]`.

---

## 9. Experiment script & notebook

**`scripts/run_hier_experiment.py`**: mirrors `scripts/run_experiment.py`.
* Imports and reuses `load_and_preprocess` and the logic of `process_bucket` from
  `scripts/run_experiment.py`. Factor out a shared helper only if that can be done **without
  editing** the baseline script; otherwise copy the bucket assembly and replace the one-hot
  dynamic build with `build_dynamic_tokens`.
* Per bucket: build `TokenSchema` → `XD2NetHier` → `Trainer` → `HierExplainer`. Write
  `predictions_*.csv` and `explanations_*.csv` (Level-2, same format as the baseline), plus
  `attention_cells_*.csv` and `attention_case_fields_*.csv`. Save `model_config_*.json`.
* Writes `results.csv` under `outputs_hier/`, in the same schema as the baseline, for direct AUC
  comparison.

**`notebooks/05_hier_explain.ipynb`**: per-case T×F joint-attention heatmap with label
annotations; α bar chart per case; global field and (field, label) attention bars; baseline vs.
hierarchical AUC per bucket; diagnostics (self-mass distribution, attention entropy).

**Optional faithfulness checks** (notebook section, not a hard requirement): deletion curve. Mask
the top-k α events (set to PAD and mark padded) and the top-k β fields (zero the token) and plot
the change in prediction against random masking. Also compute the correlation between β and
|gradient × input| on the tokenizer output.

---

## 10. Tests — `tests/hierarchical/` (pytest, CPU, tiny synthetic data)

Fixture: a schema with 2 categorical fields (cards 5, 3) and 2 numeric fields; B=4, T=6, mixed
lengths including 1 and T; `d_model=16, nhead=2`.

| # | Test | File |
|---|---|---|
| 1 | `build_dynamic_tokens`: shift by +1, PAD only where t ≥ length, numeric passthrough, float exactness | test_data |
| 2 | `TokenSchema` `to_dict`/`from_dict` round-trip; `index_to_label` maps back to original labels | test_data |
| 3 | `AttnEncoderLayer` returns attn of shape (N,H,L,L), rows sum to 1 | test_layers |
| 4 | Head-mean readout equals manual `attn.mean(1)` | test_layers |
| 5 | `cls_readout("rollout")` == `cls_readout("last")` when there is 1 layer | test_layers |
| 6 | Rollout rows sum to 1 for 3 layers | test_layers |
| 7 | β sums to 1 over fields for every real event; β = 0 at pad events | test_model |
| 8 | α sums to 1 over real events; α = 0 at padding | test_model |
| 9 | joint sums to 1 per case | test_model |
| 10 | **Mask invariance**: adding extra padded events (larger T, same lengths, arbitrary pad content) leaves `act_output`, α (real part), β (real part) unchanged (eval mode, atol 1e-5) | test_model |
| 11 | Default `forward` returns a 3-tuple; `return_attn=True` returns `HierOutput` with the documented shapes | test_model |
| 12 | Batch size 1 and `lengths=None` both work | test_model |
| 13 | All encoder params are under `model.encoder` (param count of `model.encoder` + heads == total) | test_model |
| 14 | `Trainer` runs 2 epochs on synthetic data with `XD2NetHier` without modification, and loss is finite | test_model |
| 15 | `HierExplainer.explain_attention_loader`: row count = Σ lengths × F; labels correct; JOINT sums to 1 per case; raises on caseid mismatch | test_explainer |
| 16 | Inherited `explain_loader` works and produces the baseline column schema | test_explainer |

Add `pytest` to a `[dev]` extra in `setup.cfg` only if the user approves editing it; otherwise
document `pip install pytest`.

---

## 11. Acceptance criteria

1. All tests in §10 pass: `pytest tests/hierarchical -q`.
2. `python scripts/run_hier_experiment.py --config configs/bpic2012_hier.yaml --buckets 1_10`
   completes and writes `results.csv`, prediction/explanation CSVs and attention CSVs under
   `outputs_hier/`.
3. `git diff main -- xd2net/model xd2net/explain xd2net/training xd2net/data xd2net/config.py scripts/run_experiment.py`
   is empty (baseline untouched).
4. AUC for bucket 1_10 is reported next to the baseline. There is no hard threshold, but a drop
   of more than 0.03 AUC should be investigated before the design is considered validated.
5. The notebook renders at least one per-case joint heatmap with labelled categorical cells.

---

## 12. Known baseline issues (record here; do not fix in this work)

* `Trainer.__init__` and `_train_epoch` read `cfg.save_checkpoints`, but that field exists only
  on `OutputConfig`, not on `TrainingConfig`. The script must work around this (e.g. by setting
  the attribute on the `TrainingConfig` instance before constructing `Trainer`) or fix it
  in a separate, approved change.
* `ReduceLROnPlateau(verbose=...)` is deprecated in recent PyTorch and emits a warning.
* README describes `configs/bpic2012.yaml`, but `configs/` is not tracked in git. The variant's
  YAML must be added explicitly.

---

## 13. Implementation order

1. `data.py` + tests 1–2
2. `layers.py` + tests 3–6
3. `tokenizer.py`, `model.py` + tests 7–14
4. `explainer.py` + tests 15–16
5. `config.py`, YAML, `run_hier_experiment.py`; run bucket 1_10 end-to-end
6. `05_hier_explain.ipynb`
