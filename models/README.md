# models/

Trained model artifacts saved by `scripts/train.py` (e.g. `model.joblib`).

**This folder is gitignored** by default. Options for sharing trained models:

- **Small models (<100MB):** you *can* commit them — remove the `models/*` rule from `.gitignore` if you choose to.
- **Large models (>100MB):** GitHub rejects them. Use [Git LFS](https://git-lfs.com/):

  ```bash
  git lfs install
  git lfs track "models/*.joblib"
  git add .gitattributes
  ```

- **Better for real projects:** store models in a model registry or cloud storage (MLflow, W&B, S3, Hugging Face Hub) and commit only the reference/version.
