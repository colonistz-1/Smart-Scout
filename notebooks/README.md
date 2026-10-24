# notebooks/

Exploratory Data Analysis (EDA) and experiments live here.

**Rules of thumb:**

1. **Number your notebooks** so their order is obvious: `01_eda.ipynb`, `02_baseline_model.ipynb`, `03_feature_engineering.ipynb`, ...
2. **Notebooks are for exploring, not for production.** Once a piece of code works and you use it more than once, move it into `src/` and import it — copy-pasting cells between notebooks is how projects rot.
3. **Restart & run all before committing** so the saved notebook actually runs top to bottom.
4. **Keep outputs small.** Clearing giant cell outputs before committing keeps diffs reviewable.

Because `src/` is a proper package, notebooks can reuse the exact same code as the scripts:

```python
import sys
sys.path.insert(0, "..")

from src.data import load_data
df = load_data("../data/iris.csv")
```
