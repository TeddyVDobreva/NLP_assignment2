To do:
- create CNN
- create LSTM
    - these should have the same splits, same tokenizer, same max length, same metrics.
- hyperparameter tuning
- Report Accuracy + Macro-F1 + confusion matrix for both models.
- Include learning curves (train loss + dev metric) for both models.

- Ablation (choose ONE, required)
  - Embeddings: pretrained vs random initialization
  - Max length: e.g., 64 vs 128 vs 256 tokens
  - Regularization: dropout 0 vs 0.3 (or another controlled value)
  - Capacity: hidden size small vs medium (keep all else fixed)
