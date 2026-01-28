# Movie Recommendation System

A collaborative filtering movie recommendation system using a two-phase approach:
- **Phase S1 (Item-based)**: Filters candidates using movie-movie Jaccard similarity
- **Phase S2 (User-based)**: Predicts final ratings using user-user Adjusted Cosine similarity

Supports MovieLens datasets (100K to 1M+ ratings) with multiprocessing acceleration.

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, scipy, scikit-learn, tqdm

## Usage

```bash
# Basic run (single-threaded)
python hw4.py -k 10 -t 0.8 -m 1

# Multi-threaded (8 workers)
python hw4.py -k 10 -t 0.8 -m 1 -w 8

# With reproducible results
python hw4.py -k 10 -t 0.8 -m 1 -s 42 -w 8

# Interactive mode
python hw4.py -i
```

### CLI Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-k, --neighbors` | K value for KNN | 10 |
| `-t, --train-ratio` | Training ratio (0-0.9) | 0.8 |
| `-m, --model` | 1=Jaccard+Cosine, 2=Cosine+Cosine | 1 |
| `-w, --workers` | Parallel workers | 1 |
| `-s, --seed` | Random seed | None |
| `-f, --file` | Path to ratings CSV | ratings.csv |
| `-i, --interactive` | Interactive mode | False |

## Performance

| Dataset | Workers | Total Time | Speedup |
|---------|---------|------------|---------|
| MovieLens 100K | 1 | ~9s | - |
| MovieLens 100K | 8 | ~4s | 2.3x |
| MovieLens 1M | 1 | 1:55 | - |
| MovieLens 1M | 8 | 0:25 | **4.6x** |

## Algorithm

1. **Data Loading**: Read ratings CSV, filter sparse movies (< 5 ratings)
2. **Train/Test Split**: Random split based on train ratio
3. **Similarity Matrices**: 
   - Movie-movie: Jaccard similarity (binary co-rating)
   - User-user: Adjusted Cosine similarity (mean-centered)
4. **Two-Phase Prediction**:
   - S1: Item-based filtering (threshold: 2.5)
   - S2: User-based final prediction
5. **Evaluation**: MAE, Precision, Recall, Confusion Matrix

## License

MIT
