## Building an end to end Machine Learning Project

This project trains a classification model on the wholesale customers dataset to predict the `Channel` label (`Hotel` or `Retail`).

Run the training pipeline with:

```bash
python -m src.pipeline.train_pipeline
```

After training, the saved artifacts are written to `artifacts/`:

- `raw.csv`
- `train.csv`
- `test.csv`
- `preprocessor.pkl`
- `target_encoder.pkl`
- `model.pkl`
