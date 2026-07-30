# Training Data

Place the following data files in this directory before running training:

## Files

| File | Source | Description |
|------|--------|-------------|
| `phishtank.csv` | [PhishTank](https://phishtank.org/developer_info.php) | Phishing URLs (download CSV from their developer feed) |
| `tranco_list.csv` | [Tranco](https://tranco-list.eu/) | Top 1M legitimate domains (download daily top-1m CSV) |
| `openphish.txt` | [OpenPhish](https://openphish.com/feed.txt) | Active phishing URLs (one URL per line) |

## Download Instructions

### PhishTank
```bash
curl -o phishtank.csv "https://data.phishtank.com/data/online-valid.csv"
```

### Tranco
```bash
curl -o tranco_list.csv "https://tranco-list.eu/top-1m.csv"
```

### OpenPhish
```bash
curl -o openphish.txt "https://openphish.com/feed.txt"
```

## Fallback Behavior

If data files are missing, `load_data()` generates synthetic data so the pipeline
can still be tested end-to-end.
