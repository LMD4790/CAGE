# Feature definitions

CAGE combines explicit comparison features, AST structural relations, and a contextual semantic representation from CodeBERT. LightGBM maps the explicit features to a class-support vector for the neural model.

## Explicit feature view

`cage.features.COMPARISON_FEATURE_NAMES` defines the exact zero-based column order. Support models save these names with their weights.

| Index | Name | Definition |
| --- | --- | --- |
| 0 | `jaccard` | Token-set intersection divided by union |
| 1 | `dice` | Twice the intersection divided by the sum of set sizes |
| 2 | `overlap` | Intersection divided by the smaller set size |
| 3 | `cosine` | Cosine similarity of token-frequency vectors |
| 4 | `levenshtein` | One minus normalized character edit distance after whitespace compression |
| 5 | `tfidf_cosine` | Cosine similarity using training-fitted vocabulary and IDF |
| 6–7 | `unique_tokens_code1`, `unique_tokens_code2` | Distinct lexical-token counts |
| 8 | `common_tokens` | Size of the lexical-token-set intersection |
| 9–10 | `total_tokens_code1`, `total_tokens_code2` | Lexical-token counts including repetitions |
| 11–12 | `tokens_c1`, `tokens_c2` | Counts from whitespace splitting of supplied source |
| 13 | `sim_mean` | Mean of columns 0–5 |
| 14 | `sim_std` | Sample standard deviation of columns 0–5, with `ddof=1` |
| 15–16 | `sim_max`, `sim_min` | Maximum and minimum of columns 0–5 |
| 17 | `token_ratio` | `clip(tokens_c1 / (tokens_c2 + 1e-6), 0, 10)` |
| 18 | `token_diff` | Absolute difference of whitespace-token counts |
| 19 | `struct_inter` | `cosine * (1 - levenshtein)` |

`struct_inter` measures the cosine–edit interaction. Token ratios and per-fragment counts depend on the order of `code1` and `code2`; use the same ordering for training and prediction.

The lexical tokenizer removes comments with regular expressions and substitutes string and number literals. It preserves identifier spelling. Character edit similarity instead uses whitespace-compressed source without lexical literal substitution. No additional scaling is applied to the final feature vector.

The support model stores the feature names in this order. Training and prediction check the order before using the model. Regenerate class support when the preprocessing or class definitions change.

Class support has `K` nonnegative entries summing to one. Entry `i` corresponds to label `i`, using the same class mapping throughout. Training support is out of fold; validation, test, and new inputs use the final training-fitted model.

## AST structural view

`AST_FEATURE_NAMES` contains maximum depth, log node count, named-node ratio, leaf ratio, mean children, log maximum children, normalized node-type entropy, node-type diversity, and densities for branches, loops, switches, exceptions, calls, assignments, declarations, jumps, identifiers, and literals, in that order. Log-count quantities use `log1p`.

For values `a` and `b`, define `d = min(1, abs(a-b)/(max(abs(a),abs(b))+1e-6))`. Each token contains the normalized difference `d` and its complementary similarity `1-d`, producing a `[18,2]` array per pair. The model also accepts a flat array with 18 differences followed by 18 similarities; JSONL precomputed inputs use `[18,2]`.

Both named and anonymous Tree-sitter nodes are traversed. Java and C node mappings are defined in `features.py`. Parser initialization failures raise an error. Error-recovery nodes from incomplete fragments remain part of the traversal.

## Contextual semantic view

Both fragments share the total token budget including special tokens. The first-token hidden state is the contextual representation. Optional `semantic_code1` and `semantic_code2` affect only the encoder; lexical and AST extraction use `code1` and `code2`.
