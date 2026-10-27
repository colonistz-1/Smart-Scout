# Smart-Scout

A football scouting application that groups Premier League midfielders by
statistical profile and finds players with similar contributions.

## Features

- Search among 92 eligible Premier League midfielders.
- Find statistically similar players across all clusters.
- Compare any two players side by side.
- Explore interactive radar charts with hover values.
- View per-90 statistics alongside percentile comparisons.
- Reproduce training and feature-selection experiments.

## Methodology

### Data and Preprocessing

The dataset contains Premier League player statistics for the 2025/26
season. See [`data/README.md`](data/README.md) for source details and
data acquisition instructions.

The selected model includes players listed as midfielders with at least
450 minutes played. Records missing required model features are excluded.

Counting statistics are normalized for playing time:

**Per-90 statistic = season total × 90 / minutes played**

### Selected Features

| Feature | Contribution measured |
|---|---|
| Expected goals / 90 | Scoring threat |
| Expected assists / 90 | Expected assist output |
| Tackles / 90 | Tackling Output |
| Interceptions / 90 | Interception Output |
| Key passes / 90 | Passes leading to shots |
| Accurate crosses / 90 | Completed crossing output |

Features are standardized using `StandardScaler` before fitting K-Means.

### Clustering

The selected model uses:

- **3 clusters**
- **450 minimum minutes**
- **Random state: 42**
- **K-Means initializations: 10**

The groups broadly represent:

- Creative / delivery-oriented contributions
- Defensive-minded contributions
- Contributions leading to Goals

These are provisional statistical descriptions, not fixed tactical
positions. Cluster numbers are arbitrary and should be checked after
retraining.

### Player Similarity

Players are ranked by Euclidean distance across the six standardized
features, using the scaler saved during training.

- Lower distance means greater statistical similarity.
- The selected player is excluded from their own results.
- Matches are searched across all clusters.
- Distance is not a probability or a player-quality rating.

### Radar Charts

Radar charts show percentile ranks relative to all eligible midfielders
in the loaded dataset. Percentiles are used for visualization; similarity
search continues to use standardized feature distances.

Higher percentiles indicate greater output in a metric, not necessarily
better performance.

## Results

| Model | Features | Players | Clusters | Silhouette ↑ |
|---|---:|---:|---:|---:|
| Original baseline | 4 | 92 | 3 | 0.245 |
| Selected v2 model | 6 | 92 | 3 | 0.302 |

The selected v2 model has a **Davies–Bouldin Index of 1.198** and
cluster sizes of **19, 51, and 22 players**.

### Experiments

Feature engineering explored key passes, accurate crosses, and accurate
passes per 90. Accurate final-third passes and accurate own-half passes
were excluded because both source columns contained only zeros.

Cluster-count experiments compared **k = 2 through 8**. The evaluation
metrics favored different configurations, so model selection also
considered cluster sizes and interpretability. Three clusters were
retained as a useful balance between broad profiles and additional detail.

Across ten random seeds, the six-feature, three-cluster configuration
achieved:

- Mean silhouette score: **0.296**
- Mean pairwise Adjusted Rand Index: **0.903**
- Minimum pairwise Adjusted Rand Index: **0.753**

### Minimum-Minute Comparison

| Minimum minutes | Players | Silhouette ↑ | Davies–Bouldin ↓ |
|---|---:|---:|---:|
| 450 | 92 | 0.302 | 1.198 |
| 900 | 83 | 0.329 | 1.134 |

The two configurations showed strong assignment agreement among the
83 shared players (**ARI = 0.887**). After accounting for cluster-label
changes, four shared players moved between groups.

The 450-minute threshold was retained for broader player coverage,
since qualitative inspection showed similar overall profiles.
Lower-minute player statistics should be interpreted with greater caution.

Spot checks of similarity results produced plausible matches based on
football knowledge. This is qualitative validation rather than proof of
recommendation accuracy.

## Repository Structure

```text
Smart-Scout/
├── data/               Raw and processed datasets; acquisition instructions
├── src/                Reusable Python modules
│   ├── data.py         CSV loading
│   ├── preprocessing.py
│   ├── model.py        Model construction, saving, and loading
│   ├── evaluate.py     Clustering evaluation utilities
│   ├── similarity.py   Player similarity search
│   └── utils.py        Configuration utilities
├── scripts/            Training and other command-line entry points
├── notebooks/          EDA, baseline evaluation, and experiments
├── config/             YAML experiment settings
├── models/             Saved model bundles and documentation
├── app/
│   └── app.py          Streamlit interface
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

Clone this repository, open its directory, and create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Obtain the dataset using the instructions in
[`data/README.md`](data/README.md)

## Usage

### Train the Selected Model

```bash
python scripts/train.py --config config/config_exp02.yaml
```

The selected configuration saves:

```text
data/processed/midfielder_clusters_v2.csv
models/kmeans_v2.joblib
```

The model bundle contains the fitted K-Means model and its fitted scaler.

### Launch the App

```bash
streamlit run app/app.py
```

The app requires both the processed v2 dataset and saved model bundle.

1. Search for a midfielder.
2. Review their nearest statistical matches.
3. Search for another player to compare.
4. Inspect the per-90 table and percentile radar.

### Explore the Experiments

Open the numbered notebooks in VS Code to review data exploration,
baseline evaluation, feature engineering, and player similarity.

Restart the kernel and run all cells before committing notebook changes.

## Limitations

- The model covers midfielders from one competition and season.
- Statistical similarity does not establish tactical compatibility,
  player quality, or transfer success.
- Team possession, tactical responsibilities, and set-piece duties
  influence player statistics.
- Per-90 normalization does not eliminate uncertainty from limited minutes.
- The selected features do not directly measure ball progression,
  possession control, or off-ball movement.
- Related creative features can give chance creation additional influence
  in the distance calculation.
- Clusters overlap and are sensitive to feature selection and initialization.

## Data and Artifact Handling

Raw datasets and generated outputs are excluded from version control.
The selected model bundle can be explicitly tracked for deployment.

Data redistribution must follow the source provider's terms.
Keep credentials outside version control and only load trusted joblib files.