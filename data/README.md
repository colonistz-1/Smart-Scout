# Data

Raw and processed football datasets live here. Data files are gitignored.

## Source

- Provider: StatsAPI
- Source URL: [StatsAPI](https://api.thestatsapi.com/api/football/competitions)
- Season: Premier League 2025/26

Document collection instructions and any access requirements.
Keep API keys in environment variables, never in version control.

## Preprocessing

The selected model includes midfielders with at least 450 minutes.

Six counting statistics are converted to per 90:

- Expected goals
- Expected assists
- Tackles
- Interceptions
- Key passes
- Accurate crosses

Formula: statistic × 90 / minutes played.

The processed data also contains player identifiers, names, minutes,
and assigned cluster labels. Cluster labels describe statistical
profiles rather than fixed positions or player quality.

## Regeneration

After obtaining the raw dataset, run:

```bash
python scripts/train.py --config config/config_exp02.yaml
```

The output locations are controlled by the selected configuration file.

## Deployment

Provision the processed v2 dataset separately on the hosting service.
The app requires it to run. Check the provider's terms before
redistributing data.

Missing data raises an error.