# Hierarchical Self-Attention XD2-Net — Approach

Status: implemented (`xd2net/hierarchical/`) · Branch: `feattransformer-self-attention-exp`
Companion document: [`DEV_SPEC.md`](DEV_SPEC.md)

---

## 1. Motivation

The baseline XD2-Net (`xd2net/baseline/`) takes its Level-1 (event/feature) explanations from
modules that sit **outside** the Transformer:

| Signal | Baseline source | Problem |
|---|---|---|
| α (B,T) — event importance | `TemporalAttentionPool`: a separate single-head cross-attention with a learned query over the encoder output | The encoder's own self-attention is never used for explanation. α comes from an extra add-on module, not from the network's internal attention. |
| β (B,T,D) — feature weights | `FeatureWeightingBlock`: `softmax_time(tanh(Linear(enc_out)))` | The softmax runs over **time**, not features, so β[t,·] is not "how much event *t* attends to each feature". It is also computed over one-hot columns, so most of the weight sits on columns that are zero for that event. |

The goal of this variant is for Level-1 explanations to come from the Transformer's **own
self-attention**, arranged in a hierarchy that matches how a case is built:

```
features  ──(Level A: event encoder)──▶  event representation
events    ──(Level B: case encoder)───▶  case representation  =  context vector
```

The explanation is read top-down:

* **Level B** attention says how much the case attends to each event → **α[t]**
* **Level A** attention says how much each event attends to each of its features → **β[t,f]**
* Together, **A[t,f] = α[t] · β[t,f]** says how much the case attends to feature *f* of event *t*.

Level-2 (thetas → prediction) is kept exactly as it is.

---

## 2. Baseline vs. hierarchical architecture

```
BASELINE (xd2net/baseline/model.py)
  raw_input (B,T,D_onehot+num)
    → InputProjection (+pos) → TransformerEncoder ─┬─▶ TemporalAttentionPool ─▶ α (B,T)
                                                   └─▶ FeatureWeightingBlock ─▶ β (B,T,D)
  c_t = Σ_t α·β·raw_input                         (B,D)
  [c_t, static] → ThetaHead → thetas → PredictionHead(thetas, input_exp) → p

HIERARCHICAL (xd2net/hierarchical/)
  packed tokens (B,T,F)          F = number of fields (4 for BPIC2012)
    → FieldTokenizer              (B·T, F, d)      one token per field, value-aware
    → prepend [EVT]               (B·T, 1+F, d)
    → Level A encoder (self-attn) ─▶ [EVT] output  = event rep   (B,T,d)
                                  ─▶ [EVT] attn row = β           (B,T,F)
    → + learnable position emb, prepend [CASE]      (B, 1+T, d)
    → Level B encoder (self-attn, padding-masked)
                                  ─▶ [CASE] output  = case rep    (B,d)   ← context vector
                                  ─▶ [CASE] attn row = α          (B,T)
  [case_rep, static] → ThetaHead → thetas → PredictionHead(thetas, input_exp) → p
```

The Level A encoder's weights are **shared across all events**, since every event is encoded by
the same function. For Level A we reshape `(B, T, F, d) → (B·T, F, d)` so all events in a batch
run as one large batch.

---

## 3. Tokenization: keeping label identity

### 3.1 The concern

The obvious "one token per field" design (ACTIVITY, ROLE, TASK_DURATION, TIME_ELAPSED) gives
readable β. But the *label* matters: "the event attended to ACTIVITY" says less than "the event
attended to ACTIVITY = *W_Validate application*". The other option, one token per one-hot column,
keeps label identity but creates ~30+ tokens per event, and most of them say "this event is
*not* activity X".

### 3.2 Chosen design: value-aware field tokens

Use one token per field, where the token embeds **the (field, value) pair**, in the style of
FT-Transformer / TabTransformer:

| Field kind | Token |
|---|---|
| Categorical field *f* with label *v* | `tok_f = E_field[f] + E_val_f[v]` (one `nn.Embedding(card_f + 1, d, padding_idx=0)` per field) |
| Numeric field *f* with value *x* | `tok_f = E_field[f] + x · w_f + b_f` (`w_f, b_f ∈ ℝ^d`) |

Why this is better than either option above:

1. **Label identity is kept.** Each event has exactly one active label per categorical field. So
   this design equals "one token per one-hot column, with the zero columns dropped": it keeps all
   the useful information and drops the noise.
2. **Attention depends on the label.** Keys and queries are projections of the token, which
   contains `E_val_f[v]`. So the attention a field receives depends on which label it holds. The
   model can learn that `ACTIVITY = W_Validate` deserves more attention than
   `ACTIVITY = A_SUBMITTED`.
3. **Readable explanations.** β[t, ACTIVITY] is reported together with the label, e.g.
   `event 7 → ACTIVITY = W_Validate application : 0.62`.
4. **Per-label importance is still available.** Global summaries group joint attention by
   (FIELD, VALUE_LABEL). That gives the same per-label view the one-hot design would give,
   without spreading attention over absent labels.
5. **Cheap.** 1 + 4 tokens per event instead of 1 + 30+, so Level A cost drops sharply
   (attention is quadratic in token count).

Fields are tokenized separately, so a field always means the same thing at every event.
Field-identity embeddings (`E_field`) replace positional encoding inside Level A, because
features have no order.

### 3.3 Data organisation

The baseline one-hot encodes categorical indices in
`baseline/data/features.build_dynamic_onehot`. The hierarchical variant uses the integer indices that
already exist in `vec["prefixes"][feat]`:

* Categorical indices are shifted by +1 so that **0 = PAD** (`create_index` can assign 0 to a
  real category).
* Categorical indices and numeric values are **packed into one float tensor** `(N, T, C + N_num)`
  with a fixed column layout, described by the shared `EventSchema`. Float32 stores integers up to 2²⁴
  exactly.
* Packing keeps the existing 5-tuple batch format `(raw_input, static, exp, label, length)`.
  So `EventLogDataset`, `make_dataloaders`, `Trainer` and `Explainer.explain_loader` (all in
  `xd2net/common/`) serve both variants.

### 3.4 Any event log, same model settings

The baseline data path is tied to BPIC 2012: hard-coded column names and outcome labels, roles
always discovered from `RESOURCE`, and name-specific feature aggregation. The hierarchical
variant puts all of that in a **dataset config** (one YAML per log), so one shared
model + training config can be run on many logs:

* **Core columns:** caseid, timestamp, activity, and either a role column or a resource column
  (roles are then discovered with the baseline KMeans, fitted on train only).
* **Variables are declared explicitly** as `event_cat`, `event_num`, `case_cat` or `case_num`.
  Nothing is guessed. Case-level variables are *validated* to be constant within each case, with
  a configurable tolerance.
* **Outcome** comes from either an outcome column plus a positive value, or the presence of a
  target activity. In activity mode, that activity and everything after it are cut so that
  prefixes cannot leak the label. `drop_activities` removes other outcome-revealing events,
  such as negative terminal activities.
* **Derived time features:** `TIME_SINCE_PREV` and `TIME_SINCE_START`, plus optional
  `HOUR`/`WEEKDAY`. They are computed the same way for every log.
* Every event field becomes one value-aware token. Number and type of fields vary per dataset
  and are recorded in the `EventSchema`. Nothing in the model is dataset-specific.

This pipeline now lives in `xd2net/common/data/` and **both** variants use it. Only the
per-event encoding differs: one-hot for the baseline, packed field tokens here. So
`scripts/run_experiment.py --model baseline hierarchical` compares the two on identical
splits and prefixes. Package layout: `xd2net/{common,baseline,hierarchical}`, with mirrored
modules in the two variants (DEV_SPEC §2).

---

## 4. Reading explanations from internal self-attention

### 4.1 CLS readout

Each level uses a learned summary token (`[EVT]` at Level A, `[CASE]` at Level B). The token
takes part in full self-attention. Its output vector is the level's representation, and **its
row of the attention matrix** is the explanation:

```
Level A:  attn_A ∈ ℝ^{(B·T) × H × (1+F) × (1+F)}      β_raw[t, f] = attn_A[..., 0, 1+f]
Level B:  attn_B ∈ ℝ^{B × H × (1+T) × (1+T)}          α_raw[t]    = attn_B[..., 0, 1+t]
```

This is the network's own attention, and it directly decides how the summary vector is built
(the summary is a weighted sum of value projections of the tokens).

### 4.2 Multi-head averaging

Multi-head attention is allowed. For the explanation, attention weights are **averaged over
heads**:

```
β_raw = mean_h attn_A[:, h, 0, 1:]       α_raw = mean_h attn_B[:, h, 0, 1:]
```

Per-head weights are kept in the raw output for diagnostics, but the official explanation is the
head mean.

### 4.3 CLS self-attention is blocked (default)

With unrestricted self-attention, the CLS row also contains the CLS token attending **to itself**
(column 0). During implementation this turned out to matter in practice. On one seed the
`[CASE]` token put **≈100 % of its attention on itself**. α was then just the renormalised
remainder of almost no mass: a meaningless explanation attached to an accurate model.

So by default (`cls_self_attend: false`) the CLS→CLS cell is **masked** in every layer. The
consequences:

* The CLS attention row is a distribution over the input tokens only. α and β sum to 1 without
  renormalisation.
* The attention part of the CLS update is a convex combination of the input tokens' values, so
  "how much each input went into the summary" is exactly what α/β measure. (The residual still
  carries the learned CLS embedding, which is a constant across cases.)
* Everything else in the encoder is unchanged. The other tokens still attend freely.

If `cls_self_attend: true`, the row is renormalised over the real tokens:

```
β[t,f] = β_raw[t,f] / Σ_f' β_raw[t,f']        α[t] = α_raw[t] / Σ_{t' non-pad} α_raw[t']
```

and `self_mass` is exported as a diagnostic (`EVT_SELF_ATTN`, `CASE_SELF_ATTN`, taken from the
raw last-layer head-mean attention). A large self-mass means the summary relies on its learned
prior rather than on the inputs. With the default it is 0 by construction.

### 4.3b Regularisation and attention collapse

The baseline applies L2 (1e-4) to **every** encoder parameter. On the hierarchical model this
drove the query/key projections and the LayerNorm gains towards zero. That scales every attention
logit towards zero, so α and β became **uniform**: std ≈ 1e-7 across tokens, while AUC was
unchanged. The explanation silently carried no information.

`HierTrainer` therefore keeps L2 on value/output projections, FFNs and embeddings, but **excludes
W_q, W_k, LayerNorm parameters and biases**. This is the usual "no weight decay on norms/biases"
convention, extended to the two matrices that set attention sharpness. With this change α/β are
informative (std ≈ 0.1) at the same AUC. The notebook's normalised-entropy diagnostic
(`05_hier_explain` §4) is how to spot a collapse on new data: values ≈ 1 for every case mean
uniform attention.

### 4.4 Depth: one layer vs. several

* **1 layer per level (default).** The CLS token attends to tokens that have **not yet mixed with
  each other**, so the CLS row is a direct readout of "how much of each input feature/event went
  into the summary". This is the most faithful setting and the default.
* **More than 1 layer.** By the last layer each token already contains information from the
  others, so the last-layer CLS row alone is misleading. We then use **attention rollout**
  (Abnar & Zuidema, 2020), restricted to the CLS row and computed on head-averaged attention with
  the residual connection included:

  ```
  Â_l = 0.5 · mean_h(A_l) + 0.5 · I        (row-normalised)
  R   = Â_L · Â_{L-1} · … · Â_1
  explanation = R[0, 1:]  → renormalised over the input tokens
  ```

  With one layer, rollout reduces to `0.5·A + 0.5·I`. After dropping the CLS column and
  renormalising, that gives the same β/α as the direct readout. This equivalence is a unit test.
  `attn_readout: "last"` is offered as a cheaper option for comparison.

### 4.5 Padding

* **Level B:** padded events are masked with `key_padding_mask`. The `[CASE]` token is never
  masked. α on padded positions is exactly 0.
* **Level A:** a padded event still gets tokens (PAD value embeddings are zero vectors, field
  embeddings remain). They are encoded and then thrown away: Level B masks them, and their β is
  set to 0 in the outputs. Level A needs no mask, which avoids rows where every token is masked
  (those produce NaN in softmax).
* Within a real event there are no padded fields: every field is always present.

---

## 5. Hierarchical attribution

| Quantity | Definition | Normalisation |
|---|---|---|
| Event attention α[t] | §4 Level B readout | Σ_t α[t] = 1 over real events |
| Feature attention β[t,f] | §4 Level A readout | Σ_f β[t,f] = 1 for each real event |
| Joint attention A[t,f] | α[t] · β[t,f] | Σ_{t,f} A[t,f] = 1 per case |
| Case field attention | Σ_t A[t,f] | Σ_f = 1 per case |
| Case label attention | Σ_{t : label(t,f)=v} A[t,f] | per (field, label) |
| Global field / label attention | mean over cases of the two above | — |

Joint attention is a probability distribution over (event, feature) cells, so it can be shown
directly as a T × F heatmap per case, with label annotations on the categorical cells.

---

## 6. Relation to Level-2 explanations

* `ThetaHead` and `PredictionHead` are reused **unchanged**. Only the ThetaHead input width
  changes, from `D_dynamic + D_static` to `d_model + D_static`.
* The prediction is still `sigmoid(W · (thetas ⊙ input_exp) + b)`. So the exact
  logit-contribution work specified in `docs/explainer_edits.md` applies to this model without
  changes.
* The context vector changes meaning. In the baseline, `c_t = Σ α·β·raw_input` lived in raw
  feature space. Now it is a learned latent `case_rep ∈ ℝ^{d_model}`. This does not affect the
  faithfulness of Level-2, which is defined on `input_exp`, not on the context.
* Static features are still concatenated after the case encoder (`[case_rep, static]`), so α
  describes events only.

---

## 7. Faithfulness: what the attention does and does not claim

* α/β describe **how much each input went into the summary representation** at the CLS readout.
  With one layer per level this is exact, in the sense that the summary is a convex combination of
  per-token value vectors (plus residual/FFN). It is **not** a causal statement of the form
  "removing event *t* changes the prediction by *x*".
* α/β affect the prediction only **through the thetas** (context → ThetaHead → thetas →
  prediction). Exact attribution of the prediction stays with Level-2 (thetas × weights × feature
  values).
* Recommended checks (see DEV_SPEC §9): deletion/insertion curves on the top-α events and top-β
  fields, and comparison with gradient × input on the packed tokens, to confirm empirically that
  attention follows sensitivity.
* `self_mass` (§4.3, only non-zero if `cls_self_attend: true`) and attention entropy are reported so that users can tell when attention is
  uninformative (near uniform) or not driven by the inputs.

---

## 8. Design decisions log

| Decision | Choice | Rationale |
|---|---|---|
| Baseline | Untouched; new code in `xd2net/hierarchical/` | Keep a clean A/B baseline and old checkpoints |
| Tokenization | Value-aware field tokens | Keeps label identity without one-hot noise (§3) |
| Explanation readout | CLS attention row | Uses the network's internal self-attention, as requested |
| Heads | Allowed; explanation = mean over heads | User decision |
| Depth | Default 1 layer per level; rollout if > 1 | Most faithful default; rollout handles depth (§4.4) |
| CLS self-attention | Masked by default (`cls_self_attend: false`) | Observed [CASE] self-mass ≈ 1 made α meaningless (§4.3) |
| L2 regularisation | Exclude W_q, W_k, LayerNorm, biases | Full L2 collapsed attention to uniform (§4.3b) |
| Static features | Concatenate after case encoder | Smallest change; α stays about events only |
| Level-2 | Unchanged | Keeps exact prediction attribution and explainer_edits.md work |
| Data | Packed index/value tensor + `EventSchema` | Lets the shared Dataset, Trainer and Explainer run unchanged |
| Datasets | One YAML per log, explicit variable levels, shared model config | Same model settings across logs (§3.4) |
| Outcome | `column` or `activity` mode, plus `drop_activities` | Both label sources, with leakage guards |
| Layout | `xd2net/{common,baseline,hierarchical}`, mirrored modules, one runner (`--model`) | Both models on the same pipeline and splits; baseline bugs fixed rather than worked around |
