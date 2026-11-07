# Training and evaluation

## Data preparation

Prepare training, validation, and test splits using the dataset's splitting and normalization procedures. Use the same class mapping and text preprocessing across all splits.

The split checker detects duplicate unordered pairs, source fragments shared across splits, and overlapping `group_id` values. Source identity is based on tokens, preserving literals, comments, and operators while ignoring whitespace between tokens. Function-family and project relationships are supplied through `group_id` using dataset annotations.

Training support is generated out of fold. Pairs connected by a shared source fragment or declared group stay together, including transitive connections. Each class must appear in enough independent groups for the selected fold count, and every fitting fold must contain all classes. TF-IDF is fitted only on the fitting partition of each fold.

The preparation command writes the three splits, a LightGBM model, and a manifest. The manifest records file checksums, row counts, class order, feature order, and fold assignments. Training checks file consistency and fold assignments before loading the encoder. Store the manifest with the support model, and regenerate the prepared files when inputs or preprocessing change.

For repeated runs, regenerate the support vectors and train CAGE with the chosen seed for each run.

## Model selection

`train.py` selects the checkpoint with the highest validation Macro-F1, retaining the first maximum on ties. It evaluates the test split after selection and saves the predictions in input order. The LightGBM branch is fitted on the training split.

Macro metrics use the full label set `0..K-1`. Confusion matrices use true labels for rows and predicted labels for columns. All metrics are stored as fractions; multiply by 100 for percentages.

## Saved files

| File | Contents |
| --- | --- |
| `best_model.pt` | Selected model weights and model metadata |
| `encoder_config/`, `tokenizer/` | Files needed to reload the encoder and tokenizer |
| `run_config.json` | Training arguments and input checksums |
| `history.json` | Training loss and validation metrics by epoch |
| `test_metrics.json` | Final test metrics and selected epoch |
| `test_predictions.jsonl` | Labels, predictions, and probabilities for each test pair |

The checkpoint stores the selected model, without optimizer state for resuming training. Use the matching support model for new predictions. Precomputed inputs must match their preparation manifest.

## Common errors

| Message | What to check |
| --- | --- |
| Duplicate pair or shared source | Review the input pairs and split construction. |
| Missing class in an OOF fold | Check group coverage and the requested fold count. |
| Missing or mismatched manifest | Regenerate the prepared files and keep the manifest with the support model. |
| Support checksum mismatch | Use the LightGBM model from the same training run. |
| Empty or invalid semantic text | Supply nonempty text or omit the optional field. |
| Existing output directory | Choose a new output directory. |
| Encoder download unavailable | Supply a local encoder and tokenizer directory. |

See [Feature definitions](FEATURES.md) for the exact feature order and formulas.
