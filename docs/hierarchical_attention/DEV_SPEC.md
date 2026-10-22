# Hierarchical Self-Attention XD2-Net — Development Spec

Status: implemented · Branch: `feattransformer-self-attention-exp`
Rationale and math: [`APPROACH.md`](APPROACH.md)

---

## 1. Scope

* `xd2net/hierarchical/` implements:
  * the two-level self-attention model;
  * value-aware tokenization;
  * attention extraction;
  * hierarchical explanations.
* `xd2net/common/` holds what both variants share:
  * the **dataset-agnostic event-log pipeline**, so the same model/training settings run on many
    logs;
  * the Trainer, the Explainer base, the Level-2 heads and the model registry.
* `xd2net/baseline/` is the original Transformer + TemporalAttentionPool model. It has the same
  module set as `hierarchical/` and runs on the same pipeline.
* The issues found in the original baseline code are fixed (§13).
* Level-2 (ThetaHead / PredictionHead) is unchanged. The work in `docs/explainer_edits.md`
  applies to `common/explainer.py`.

---

## 2. File layout

```
xd2net/common/
├── config.py        TrainingConfig, ExperimentConfig, DatasetConfig (+ ColumnsConfig, OutcomeConfig,
│                    TimeFeatureConfig, NormConfig, PrefixConfig), load_experiment_config(path, model_cfg_cls),
│                    load_dataset_config, save_yaml
├── data/
│   ├── eventlog.py      load_event_log(): read, rename, sort, type, drop, time features, outcome,
│   │                    case-level validation
│   ├── preprocessing.py split_train_test, RoleDiscovery, create_index, normalize_events,
│   │                    extract_prefixes, balance_dataset, reformat_events, vectorize, trace_stats
│   ├── features.py      EventSchema, build_event_schema(), build_static_input(), build_exp_input()
│   ├── pipeline.py      PreparedLog, prepare_dataset(), resolve_prefix_lengths(), build_bucket()
│   └── dataset.py       EventLogDataset, make_dataloaders, make_test_loader
├── heads.py         ThetaHead, PredictionHead
├── masks.py         make_padding_mask
├── metrics.py       evaluate_predictions, feature_agreement
├── training.py      Trainer (+ `_l2_penalty` hook)
├── explainer.py     Explainer (Level-2 tables, attention-cell table, global summaries)
└── registry.py      ModelSpec, available_models(), get_spec(name)
xd2net/hierarchical/                     xd2net/baseline/
├── config.py        HierModelConfig      ├── config.py        BaselineModelConfig
├── data/features.py build_dynamic_tokens ├── data/features.py build_dynamic_onehot, dynamic_feature_names
├── layers.py        AttnEncoder(Layer),  ├── layers.py        InputProjection, EventEncoder,
│                    cls_attn_mask,       │                    TemporalAttentionPool,
│                    attention_rollout,   │                    FeatureWeightingBlock
│                    cls_readout          │
├── tokenizer.py     FieldTokenizer       │
├── model.py         XD2NetHier, HierOutput ├── model.py      XD2Net
├── training.py      HierTrainer          ├── training.py      BaselineTrainer
├── explainer.py     HierExplainer        ├── explainer.py     BaselineExplainer
└── spec.py          SPEC                 └── spec.py          SPEC
configs/models/{baseline,hierarchical}.yaml   model + training settings per variant
configs/datasets/bpic2012.yaml                BPIC 2012
configs/datasets/_template.yaml               annotated template for new logs
scripts/run_experiment.py                     multi-model, multi-dataset runner
notebooks/01–04                               per-(MODEL, DATASET_CONFIG) prepare/train/evaluate/explain
notebooks/05_hier_explain.ipynb               visualisation of hierarchical runner outputs
tests/  conftest.py, synthetic.py, test_registry.py, common/, baseline/, hierarchical/
```

**Plug-in contract (`ModelSpec`).** A variant provides:
* a model config class;
* a model class with `get_config()` / `from_config()` and the forward
  `(raw_input, static, exp, lengths) -> (thetas, prob, alpha)`, with its regularised parameters
  under `model.encoder`;
* `encode_dynamic(vec, schema, max_len) -> input_dynamic`, the only data step that differs;
* `build_model(bucket, cfg, dataset_name)`;
* a trainer class and an explainer class;
* the path of its default config.

---

## 3. Configuration

### 3.1 Model config (one per variant, shared by every dataset) — `configs/models/<model>.yaml`

```yaml
# configs/models/hierarchical.yaml
model:     {d_model: 64, nhead: 4, event_layers: 1, case_layers: 1, dim_feedforward_mult: 4,
            dropout: 0.15, attn_readout: rollout, cls_self_attend: false, dense_1_mult: 1.5}
training:  {optimizer: Adam, learning_rate: 0.01, batch_size: 256, epochs: 100, val_split: 0.15,
            early_stopping_patience: 10, lr_reduce_patience: 10, lr_reduce_factor: 0.5,
            lr_min: 1.0e-7, l2_reg: 0.0001, save_checkpoints: true, seed: 42, device: auto}
workspace_dir: ./outputs
# configs/models/baseline.yaml
model:     {d_model: 140, nhead: 4, num_encoder_layers: 1, dim_feedforward_mult: 4,
            dropout: 0.15, dense_1_mult: 1.5}
training:  {... same keys ...}
```

* `TrainingConfig` defines `l2_reg`, `save_checkpoints` and `seed`. `l2_reg` moved there from the
  baseline model config.
* `load_experiment_config(path, spec.model_config_cls)`: the variant's dataclass validates the
  `model` section, so unknown keys are rejected.
* Validation:
  * hierarchical: `d_model % nhead == 0`, `attn_readout ∈ {rollout, last}`, layers ≥ 1;
  * baseline: `d_model % nhead == 0`, layers ≥ 1.

### 3.2 Dataset config (one per event log) — see `configs/datasets/_template.yaml`

| Key | Meaning |
|---|---|
| `name`, `path` | output folder name; `.csv` or `.parquet` |
| `columns.{caseid,timestamp,activity}` | source column names |
| `columns.role` / `columns.resource` | a role column, **or** `role: null` + `resource` (roles discovered by KMeans, fit on train) |
| `timestamp_format`, `event_order` | parsing format; `timestamp` (stable sort, ties keep file order) or `file` |
| `drop_activities` | activities removed before prefixing (outcome-revealing terminal events) |
| `event_cat`, `event_num`, `case_cat`, `case_num` | **explicit** variable levels (nothing is inferred) |
| `case_level_tolerance` | share of cases allowed to violate case-level constancy (default 0) |
| `num_agg` | exp-input aggregation per numeric event field (`mean` default, `TIME_SINCE_START: max`) |
| `outcome` | `{mode: column, column, positive, labels}` or `{mode: activity, activities, labels}` |
| `time_features` | `since_prev`, `since_start` (numeric), `hour`, `weekday` (categorical), `unit` |
| `norm` | `default` (`max` = z-score, or `lognorm`) and per-field `overrides` |
| `prefix` | `lengths: auto` or a list; `min_len`, `max_len`, `quantile`, `min_cases` |
| `test_split`, `balance_train` | as in the baseline |

* Unknown keys raise an error, so typos are caught.
* Reserved names (`CASEID`, `ACTIVITY`, `ROLE`, `TIME_SINCE_*`, …) cannot be used as variable
  names.
* Event field order in the model: `[ACTIVITY, ROLE, *event_cat, (HOUR), (WEEKDAY)]` followed by
  `[*event_num, (TIME_SINCE_PREV), (TIME_SINCE_START)]`.

---

## 4. Data pipeline — `xd2net/common/data/` (shared by both variants)

### 4.1 `load_event_log(ds, df=None)`
1. Select the configured columns only, then rename them to `CASEID, TIMESTAMP, ACTIVITY, ROLE |
   RESOURCE` (outcome column → internal `_OUTCOME_RAW`). Missing columns raise and list the
   available ones.
2. Parse timestamps. Unparseable rows are dropped with a warning. `CASEID` is cast to `str`. Rows
   are sorted per `event_order` with a mergesort (stable).
3. Categoricals: NaN or empty → `"none"`, then `str`. Numerics: `to_numeric(errors="coerce")`,
   with coerced values logged.
4. Drop `drop_activities`, then derive time features.
5. Outcome:
   * `column`: the per-case value must be constant (within tolerance). Positive = equal to
     `positive`, compared both as raw values and as strings.
   * `activity`: positive = the case contains a target activity. The first target occurrence and
     everything after it are removed. Cases left empty are dropped.
   * In both modes `OUTCOME` is the label string, so the baseline `extract_prefixes` and
     `balance_dataset` work. A single resulting class raises.
6. Case-level validation: `nunique` per case for each `case_*` variable. If the share of cases
   that vary exceeds the tolerance, raise with examples. Otherwise broadcast the first non-null
   value per case.
7. `ACTIVITY_INDEX = cumcount + 1`.

### 4.2 `prepare_dataset(ds, seed, df=None) -> PreparedLog`
* `split_train_test`.
* Roles: use the role column, or apply `RoleDiscovery` fit (train) / transform (test). That needs
  a `MILESTONE` alias.
  * KMeans runs on binary resource × activity profiles of train resources.
  * On any frame, seen resources keep their train role. Missing and unseen resources get
    `"none"`.
  * No events are dropped.
  * Integer-valued float ids (from a column with NaN) match their int form.
* Numeric imputation with train medians. Normalisation is grouped by method, calling
  `normalize_events` once per group (train fit, test transform).
* Categorical indexes from train with `create_index(reserve_zero=True)`. Unseen and missing
  values map to 0 = `"none"`.
* `PreparedLog.summary()` gives a JSON description (fields, cardinalities, case-length
  quantiles).

### 4.3 `resolve_prefix_lengths(prepared)`
An explicit list, or `auto` = `range(min_len, min(max_len, quantile(train case lengths)) + 1)`.

### 4.4 `build_bucket(prepared, n, encode_dynamic, seed) -> dict | None`
* `extract_prefixes`. The bucket is skipped if there are fewer than `min_cases` train cases, no
  test cases, or a single train class.
* Balancing: `balance_dataset` always undersamples the **actual minority label**. It used to take
  the positive label, which breaks when positive is the majority class.
* `reformat_events` (sequences reversed, most recent event first) → `trace_stats` → `vectorize`.
* Inputs:
  * `encode_dynamic` (variant-specific):
    * hierarchical `build_dynamic_tokens`: packed `(N, T, n_cat + n_num)`, category index + 1,
      0 = PAD at `t ≥ length`.
    * baseline `build_dynamic_onehot`: `(N, T, Σ cardinalities + n_num)`, all-zero rows at
      `t ≥ length`.
  * `build_static_input`.
  * Generic `build_exp_input`: categorical counts per label with **padding excluded**
    (`"Occurrence of F=label"`); numeric aggregations (`"Mean F"`, `"Max F"`, …); then static
    fields.
* Train and test share the same indexes, so the feature names always match. The old
  `align_exp_dims` zero-padding is gone.
* Returns: `bucket_key`, `inputs_/outputs_/lengths_/caseids_{train,test}`, `exp_dim`, `max_len`,
  `feature_names`, `dim_dynamic`, `dim_static`, `event_schema`, `n_train`, `n_test` and
  `dataset`.
* Everything except `inputs_*["input_dynamic"]` / `dim_dynamic` is identical across variants
  (tested).

### 4.5 `EventSchema` (`common/data/features.py`)
`cat_fields`, `num_fields`, `cardinalities` (excluding PAD), `index_to_label` (shifted index →
label). It has `to_dict` / `from_dict` for JSON. Packed token indices must be `< 2²⁴` (exact in
float32). Both models store it (`model.schema`), so the explainer can decode event values.

---

## 5. Layers — `xd2net/hierarchical/layers.py`

* `AttnEncoderLayer`: pre-norm MHA (`need_weights=True, average_attn_weights=False`) + GELU FFN.
  Takes `key_padding_mask` and `attn_mask`. Returns `(out, attn (N,H,L,L))`.
* `AttnEncoder`: a stack of layers plus a final LayerNorm. Returns `(out, [attn per layer])`.
* `cls_attn_mask(L, cls_self_attend)`: an `(L, L)` bool mask blocking the CLS→CLS cell, or
  `None` when self-attention is allowed.
* `attention_rollout(attns)`: `Â = 0.5·mean_h(A) + 0.5·I`, row-normalised; `R = Â_L ⋯ Â_1`.
* `cls_readout(attns, mode, token_mask)`:
  * Row 0 from `last` (head mean) or `rollout`.
  * Drop column 0, zero the padded tokens, renormalise.
  * `self_mass` = raw last-layer head-mean `A[0,0]`. It is never taken from rollout, because the
    identity term would make it ≥ 0.5.

## 6. Tokenizer — `FieldTokenizer(schema, d_model)`
* Categorical token: `E_field[f] + E_val_f[v]` (`padding_idx=0`).
* Numeric token: `E_field[f] + x·w_f + b_f`.
* Shape: `(B, T, n_cat+n_num) → (B, T, F, d)`.

## 7. Model — `XD2NetHier(schema, dim_static, exp_dim, max_len, cfg, dataset_name)`

* `encoder = ModuleDict(tokenizer, event, case)`. Every encoder parameter lives here.
* **Level A** `EventLevelEncoder`: `(B·T, 1+F, d)` with `[EVT]`. No padding mask, because padded
  events are discarded later. CLS self-cell masked by default. Outputs: event rep, β, and the
  `[EVT]` self-mass.
* **Level B** `CaseLevelEncoder`: event reps + position embedding (zeroed at padding) +
  `[CASE]`. Key-padding mask, CLS self-cell masked by default. Outputs: case rep, α, and the
  `[CASE]` self-mass.
* `ThetaHead(d_model + dim_static)` and `PredictionHead(exp_dim)` from the baseline.
* `forward(raw_input, static, exp, lengths=None, return_attn=False)`:
  * default → `(thetas, prob, alpha)`, compatible with the baseline Trainer/Explainer;
  * `return_attn=True` → `HierOutput(thetas, act_output, alpha, beta, joint, evt_self_attn,
    case_self_attn, event_attn_raw, case_attn_raw)`. β and evt_self are zeroed at padded events.
* `get_config()` / `from_config()` rebuild the module from JSON plus a state_dict.

## 8. Trainer — `common.training.Trainer` + `HierTrainer` / `BaselineTrainer`

The shared `Trainer` provides:
* the train loop and validation (AUC);
* `ReduceLROnPlateau`;
* early stopping;
* `_save_checkpoint`, which keeps only the current best file;
* `load_best_model`, with `weights_only=True`. When checkpoints are off, it restores an in-memory
  copy of the best state.

The L2 term is `cfg.l2_reg * _l2_penalty()`:
* `Trainer._l2_penalty` (used by `BaselineTrainer`): every `model.encoder` parameter, as in the
  original.
* `HierTrainer._l2_penalty`: **excludes attention W_q/W_k, LayerNorm parameters and biases**.
  Full L2 collapsed attention to uniform (APPROACH §4.3b).

## 9. Explainer — `common.explainer.Explainer` + `HierExplainer` / `BaselineExplainer`

* Shared:
  * `explain_loader` / `global_explanation` (Level-2) — vectorised, same output as before;
  * `get_temporal_attention`;
  * `explain_attention_loader`;
  * the static `global_attention`.
* Variant hooks:
  * `_attention_batch` returns α and, if the model has them, β and the self-masses (otherwise
    `None` → NaN columns).
  * `_decode_events` turns dynamic-input rows into shifted indices / values:
    * hierarchical: identity;
    * baseline: argmax per one-hot block + 1, numerics as is.
* `HierExplainer.get_hierarchical_attention(...) -> {alpha, beta, joint, evt_self_attn, case_self_attn}`.
* `explain_attention_loader(loader, caseids, bucket) -> (df_cells, df_case_fields)`. The schema
  comes from `model.schema`.
  * **df_cells**: one row per (case, real event, field):
    `CASEID, EVENT_POS (0 = most recent), EVENT_INDEX (0 = first), EVENT_ACTIVITY, EVENT_ROLE,
    FIELD, VALUE_LABEL, VALUE_NUM, EVENT_ATTN (α), FEATURE_ATTN (β), JOINT_ATTN (α·β),
    EVT_SELF_ATTN, BUCKET`.
    Rows are ordered by loader case order, then chronological event, then schema field order.
  * **df_case_fields**: `CASEID, FIELD, CASE_FIELD_ATTN (Σ_t α·β), CASE_SELF_ATTN, BUCKET`.
  * Raises on a caseid-count or field-count mismatch. Warns on a non-sequential loader.
  * Baseline: β, joint and self-mass columns are NaN. α comes from the TemporalAttentionPool.
* `global_attention(df_cells, by="field" | "field_label")`: mean per-case attention (absent
  counts as 0) plus `N_CASES`. Only meaningful when `JOINT_ATTN` is present.

## 10. Runner — `scripts/run_experiment.py`

```
python scripts/run_experiment.py --model baseline hierarchical \
    --datasets configs/datasets/a.yaml configs/datasets/b.yaml \
    [--model-config m1.yaml m2.yaml] [--buckets 1_5 1_10] [--seed 42] [--epochs N] \
    [--workspace DIR] [--fail-fast]
```
* `--model-config` defaults to `configs/models/<model>.yaml`. `--seed` / `--epochs` override
  every model's training config. The workspace and the data seed come from the first model
  unless `--workspace` / `--seed` are given.
* For each dataset:
  * `prepare_dataset` runs **once** and is shared by all models.
  * Under `<workspace>/<model>/<dataset>/`, save `dataset_config.yaml`, `experiment_config.yaml`
    and `dataset_summary.json` (including the resolved prefix lengths).
  * For each bucket and each model:
    * `build_bucket(..., spec.encode_dynamic)`, seed, `spec.build_model`, `spec.trainer_cls`;
    * best checkpoint plus `model_config_<bucket>.json`;
    * `predictions_` / `explanations_` (Level-2);
    * `attention_cells_` / `attention_case_fields_` (Level-1), plus `attention_global_` when β
      exists;
    * metrics.
* Per-(model, dataset) `results.csv` and a combined `<workspace>/results_all.csv` (`model,
  dataset, bucket, n_train, n_test, epochs, accuracy, precision, recall, f1, roc_auc`).
* Metrics are NaN when the test bucket has a single class. A failing dataset is logged and the
  runner continues, unless `--fail-fast` is given.

## 11. Notebooks

* **01–04:**
  * Parameters `MODEL`, `DATASET_CONFIG`, `MODEL_CONFIG`.
  * 01 runs `prepare_dataset` + `build_bucket` and pickles the buckets to
    `<workspace>/<MODEL>/<dataset>/artifacts/`.
  * 02 trains via the spec and writes the checkpoint + `model_config_<bucket>.json`.
  * 03 and 04 rebuild models with `spec.model_cls.from_config`.
* **05 (`05_hier_explain.ipynb`)** reads runner outputs. Parameters: `WORKSPACE`, `MODEL`,
  `DATASET`, `BUCKET`, `CASE_ID`. It shows:
* AUC per bucket per (model, dataset), and a side-by-side AUC table of the models for the
  selected dataset;
* a per-case T×F joint heatmap with label annotations, next to α;
* the β table;
* global attention by field and by (field, label);
* α by event index;
* diagnostics: normalised entropy of α/β (≈ 1 everywhere = collapsed attention) and CLS
  self-mass;
* Level-2 theta importances.

## 12. Tests — `pytest tests -q`

Layout: `tests/common/` (data, fixes), `tests/baseline/`, `tests/hierarchical/` and
`tests/test_registry.py`.

* **Registry** (for every model):
  * synthetic bucket → model → 1 epoch → `from_config` round-trip → Level-2 and attention-cell
    tables (fixed column schema, α sums to 1);
  * both models get identical buckets apart from `input_dynamic`.
* **Fixed issues** (`tests/common/test_fixes.py`):
  * RoleDiscovery keeps every event, maps NaN/unseen to `none` and handles numeric/float ids;
  * actual-minority balancing;
  * padding-free counts;
  * only the best checkpoint is kept, loaded with `weights_only`;
  * in-memory best state when checkpoints are off;
  * seeded shuffle;
  * the `TrainingConfig` fields.
* **Baseline:**
  * one-hot layout with zero padded rows;
  * shapes;
  * α sums to 1 and is 0 on padding;
  * mask invariance;
  * config round-trip;
  * trainer + single checkpoint;
  * L2 over all encoder parameters.
* **Data:**
  * config validation (unknown keys, reserved/duplicate names, field order);
  * sorting with ties and `event_order: file`;
  * time features;
  * missing values and coercion;
  * column outcome; `drop_activities`; activity outcome with the leakage cut;
  * case-level validation and tolerance;
  * missing-column error; single-class error;
  * unseen labels → `none`; imputation; role discovery path;
  * auto prefix lengths;
  * bucket shapes and balancing; bucket skipping;
  * token shift/PAD; schema round-trip;
  * exp-input padding exclusion and all aggregations;
  * every shipped config parses.
* **Layers:**
  * per-head attention shape and row sums;
  * the CLS mask zeroes the self cell;
  * head-mean readout;
  * rollout == last for 1 layer; rollout row sums;
  * the token mask.
* **Model:**
  * output types and shapes;
  * β/α/joint normalisation and zeros on padding;
  * self-mass 0 by default; the `cls_self_attend` option;
  * **mask invariance** (extra junk padding leaves prob/α/β unchanged) for rollout/last at
    depths 1–3;
  * batch size 1 and `lengths=None`;
  * parameter ownership under `encoder`;
  * config round-trip;
  * trainer run plus checkpoint;
  * the L2 penalty skips Q/K/LayerNorm but includes V.
* **Explainer:**
  * row counts;
  * per-case and per-event sums;
  * the EVENT_POS/EVENT_INDEX relation;
  * labels vs values; activity column consistency;
  * case order;
  * global sums;
  * caseid mismatch error;
  * inherited Level-2 output schema.

`tests/synthetic.py` generates a log with non-standard column names, case- and event-level
variables, and both outcome encodings. Run as a script, it writes a CSV and a dataset YAML for
smoke runs.

## 13. Baseline issues (fixed in the restructure)

| # | Issue in the original baseline code | Fix |
|---|---|---|
| 1 | `Trainer.__init__` passed `verbose=` to `ReduceLROnPlateau` (TypeError on current PyTorch) | Removed |
| 2 | `Trainer` read `cfg.save_checkpoints` / `cfg.l2_reg`, which `TrainingConfig` lacked | Added to `TrainingConfig` with `seed`; `l2_reg` removed from the model config |
| 3 | `RoleDiscovery.transform` failed on pandas ≥ 3; its inner `merge` dropped events with NaN resources; seen test resources were re-clustered from their test profile; unseen ones became their raw id (a new category) | Role map from train; seen resources keep their train role; NaN/unseen → `"none"`; no rows dropped; float ids normalised |
| 4 | `load_and_preprocess` sorted by a hard-coded `"Complete Timestamp"` | Replaced by `load_event_log` (`event_order` per dataset) |
| 5 | `build_exp_input` counted padded positions as category-0 occurrences | Generic builder masks by `lengths` |
| 6 | The one-hot dynamic input set the category-0 bit at padded steps | `build_dynamic_onehot` zeroes padded rows |
| 7 | `balance_dataset` was called with the positive label as "minority" | `balance_dataset` always uses the actual minority (a wrong hint logs a warning) |
| 8 | Target names `A_DECLINED/A_APPROVED` hard-coded | Taken from `outcome.labels` |
| 9 | `configs/bpic2012.yaml`, referenced by the README and notebooks, wasn't tracked | `configs/datasets/bpic2012.yaml` + `configs/models/baseline.yaml` |
| 10 | `torch.load` without `weights_only`; every improving epoch left a checkpoint; with checkpoints off, `load_best_model` returned the last epoch | `weights_only=True`, single best file, in-memory best state |
| 11 | Seeds not set; the training shuffle was unseeded | The runner seeds torch/numpy/random; `make_dataloaders` seeds the shuffle |
| 12 | Exp feature names special-cased `ROLE` / `TIME_ELAPSED` | Generic names; `num_agg` per dataset (BPIC: `TIME_ELAPSED: max`) |
| 13 | `pin_memory=True` on CPU-only machines (warning) | Enabled only with CUDA |

Unchanged on purpose: the baseline architecture and parameter names (old `state_dict`s load
strictly into `baseline.model.XD2Net`), and its L2 over all of `model.encoder`.

Because of fixes 3, 5 and 6, baseline results on BPIC are not bit-identical to the pre-restructure
code. One example: 8,120 BPIC events with a missing resource now get `ROLE=none` instead of being
clustered as a pseudo-resource.
