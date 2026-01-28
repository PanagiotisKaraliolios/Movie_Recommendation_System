# Movie Recommendation System

A collaborative filtering movie recommendation system using a two-phase approach:
- **Phase S1 (Item-based)**: Filters candidates using movie-movie Jaccard similarity
- **Phase S2 (User-based)**: Predicts final ratings using user-user Adjusted Cosine similarity

Supports MovieLens datasets (100K to 25M ratings) with automatic download and multiprocessing acceleration.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, scipy, scikit-learn, tqdm

## Usage

### Dataset Selection

The system can automatically download and prepare MovieLens datasets:

```bash
# Use MovieLens 100K (downloads if not present)
python hw4.py -d 100k

# Use MovieLens 1M
python hw4.py -d 1m -w 8

# Use MovieLens 10M (requires ~8GB RAM)
python hw4.py -d 10m -w 8

# Use local ratings.csv file
python hw4.py -d local

# Interactive mode (prompts for dataset selection)
python hw4.py -i
```

Available datasets:
| Dataset | Size | RAM Required |
|---------|------|--------------|
| `100k` | ~100K ratings | ~1GB |
| `1m` | ~1M ratings | ~2GB |
| `10m` | ~10M ratings | ~8GB |
| `25m` | ~25M ratings | ~32GB |
| `local` | Use ratings.csv | varies |

### Full Examples

```bash
# Basic run with dataset selection
python hw4.py -d 1m -k 10 -t 0.8 -m 1

# Multi-threaded (8 workers)
python hw4.py -d 1m -k 10 -t 0.8 -m 1 -w 8

# With reproducible results
python hw4.py -d 1m -k 10 -t 0.8 -m 1 -s 42 -w 8

# Interactive mode
python hw4.py -i
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-d, --dataset` | Dataset: 100k, 1m, 10m, 25m, local | interactive |
| `-k, --neighbors` | K value for KNN | 10 |
| `-t, --train-ratio` | Training ratio (0-0.9) | 0.8 |
| `-m, --model` | 1=Jaccard+Cosine, 2=Cosine+Cosine | 1 |
| `-w, --workers` | Parallel workers | CPU count - 1 |
| `-s, --seed` | Random seed | None |
| `-f, --file` | Direct path to CSV (overrides -d) | None |
| `-i, --interactive` | Interactive mode | False |

## Performance

| Dataset | Workers | Total Time | Speedup |
|---------|---------|------------|---------|
| MovieLens 100K | 1 | ~9s | - |
| MovieLens 100K | 8 | ~4s | 2.3x |
| MovieLens 1M | 1 | 1:55 | - |
| MovieLens 1M | 8 | 0:25 | **4.6x** |

## Algorithm

1. **Dataset Preparation**: Download and convert dataset if needed
2. **Data Loading**: Read ratings CSV, filter sparse movies (< 5 ratings)
3. **Train/Test Split**: Random split based on train ratio
4. **Similarity Matrices**: 
   - Movie-movie: Jaccard similarity (binary co-rating)
   - User-user: Adjusted Cosine similarity (mean-centered)
5. **Two-Phase Prediction**:
   - S1: Item-based filtering (threshold: 2.5)
   - S2: User-based final prediction
6. **Evaluation**: MAE, Precision, Recall, Confusion Matrix

## License

MIT
