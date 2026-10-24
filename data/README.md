# data/

Raw and processed datasets live here. **This folder is gitignored** — data files are never committed (they are large, may be private, and can always be regenerated or re-downloaded).

Instead, commit *instructions* for getting the data. Document here:

- Where the data comes from (URL, database, API)
- How to download or regenerate it
- What each file/column means

A common convention as projects grow:

```
data/
├── raw/          Original, immutable data — never edit these files
├── processed/    Cleaned data produced by your code
└── external/     Third-party reference data
```

> The example pipeline in this template falls back to scikit-learn's built-in Iris dataset when `data/iris.csv` is missing, so everything runs before you add any data.
