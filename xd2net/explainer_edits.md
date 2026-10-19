Refactor xd2net/explain/explainer.py to provide prediction-faithful,
business-readable explanations while preserving backwards compatibility.

Requirements:

1. Preserve the existing Explainer constructor, explain_loader(),
   get_temporal_attention(), and global_explanation() entry points.

2. In explain_loader(), export separate columns for:
   - THETA
   - MODEL_WEIGHT
   - EFFECTIVE_COEFFICIENT = THETA * MODEL_WEIGHT
   - FEATURE_VALUE_MODEL = the transformed value passed to the model
   - LOGIT_CONTRIBUTION = THETA * MODEL_WEIGHT * FEATURE_VALUE_MODEL
   - ABS_CONTRIBUTION
   - CONTRIBUTION_SHARE_ABS
   - CONTRIBUTION_DIRECTION
   - CONTRIBUTION_RANK_ABS

3. Do not call THETA * MODEL_WEIGHT a contribution.
   Retain FEATURE_IMPORTANCE only as a deprecated backward-compatible alias
   for EFFECTIVE_COEFFICIENT.

4. Support optional raw feature values and business metadata:
   - raw_feature_values aligned with caseids and feature_names
   - feature_metadata keyed by feature name
   - optional display name, unit, baseline raw value, and baseline model value

5. If raw values are supplied, export:
   - FEATURE_VALUE_RAW
   - FEATURE_DISPLAY
   - FEATURE_UNIT
   - BASELINE_VALUE_RAW where available

6. If baseline model-space values are supplied, export:
   FROZEN_THETA_DEVIATION =
       THETA * MODEL_WEIGHT *
       (FEATURE_VALUE_MODEL - BASELINE_VALUE_MODEL)

   Explicitly document that this freezes the observed THETA and is not an
   exact counterfactual effect.

7. Add exact reconstruction diagnostics:
   - sum all LOGIT_CONTRIBUTION values per case
   - add prediction-head bias if present
   - compare reconstructed logits with the model logits
   - expose RECONSTRUCTED_LOGIT and RECONSTRUCTION_ERROR in df_predictions
   - optionally raise if error exceeds a configurable tolerance

8. Handle logits/probabilities carefully:
   - determine whether act_output contains logits or probabilities from the
     actual PredictionHead implementation
   - do not apply sigmoid twice
   - if the current model returns only probabilities, either:
       a. obtain logits from a supported model output/API, or
       b. reconstruct logits from contribution sum plus bias and verify that
          sigmoid(reconstructed_logit) matches the returned probability

9. Account for prediction-head bias:
   - export BASE_VALUE or PREDICTION_BIAS in df_predictions
   - explain that feature contributions reconstruct the logit only after
     adding this bias

10. Guard against zero total contribution:
    CONTRIBUTION_SHARE_ABS should be zero when the sum of absolute
    contributions is zero, never NaN or infinity.

11. Support global summaries based on a selectable metric:
    - mean absolute logit contribution, recommended default
    - median absolute logit contribution
    - mean signed logit contribution
    - mean absolute contribution share
    - legacy effective coefficient

12. For global explanations, do not average signed contributions by default,
    because positive and negative effects can cancel.

13. Add validation for:
    - caseids count equals number of model outputs
    - len(feature_names) equals exp_dim
    - final prediction-head input width equals exp_dim
    - raw feature array shape matches model feature array shape
    - baseline vectors match exp_dim
    - loader order is deterministic or caseids are included in each batch

14. Add unit tests covering:
    - exact logit reconstruction including bias
    - negative model weights
    - negative standardised feature values
    - zero-contribution rows
    - correct absolute shares summing to one for non-zero rows
    - raw/model value alignment
    - baseline alignment
    - output shape for batch size one
    - DataLoader shuffling/order protection
    - CPU and accelerator-safe tensor conversion
