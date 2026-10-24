# outputs/

Everything your runs *produce*: metrics, figures, logs, predictions.

**This folder is gitignored** — outputs are reproducible by re-running the code, so they don't belong in version control (and figures/logs change on every run, which creates noisy diffs).

After running `python scripts/train.py` you will find:

```
outputs/
├── metrics.json                  Accuracy, F1, ... for the run
└── figures/
    └── confusion_matrix.png      Evaluation figure
```

Tip: when you want to *share* a key result (e.g. the headline figure for the README), copy it somewhere tracked — or better, report the numbers in the README's Results section.
