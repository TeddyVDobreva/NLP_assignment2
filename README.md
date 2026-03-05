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



To do list part 2:
1. tokenizer
2. CNN architecture (+ LSTM)
3. decide on ablation (max len easiest)
4. early stopping
5. error analysis (eventually)