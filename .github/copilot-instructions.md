# Copilot Instructions for Movie Recommendation System

## Project Overview

**Collaborative filtering movie recommendation system** using a two-phase approach:
- **Phase S1 (Item-based)**: Filters candidates using movie-movie similarity (threshold: 2.5)
- **Phase S2 (User-based)**: Predicts final ratings using user-user similarity

Supports MovieLens datasets (100K to 1M+ ratings) with multiprocessing acceleration.

## Architecture

### Data Flow
```
ratings.csv → read_ratings() → filter_sparse_movies() → train_test_split()
           → build_similarity_matrices() → build_train_data_index()
           → run_two_phase_prediction() → evaluate_predictions()
```

### Key Components in `hw4.py`

| Component | Purpose |
|-----------|---------|
| `SimilarityMatrices` | Dataclass holding precomputed numpy arrays + index mappings |
| `TrainDataIndex` | Pre-indexed training data for O(1) lookups (grouped by user/movie) |
| `EvaluationMetrics` | Dataclass for MAE, precision, recall, confusion matrix |
| `build_similarity_matrices()` | Vectorized similarity using scipy/sklearn |
| `_init_s1_worker/_init_s2_worker` | Process initializers for multiprocessing |
| `_predict_item_based_worker` | Item-based CF worker (Phase S1) |
| `_predict_user_based_worker` | User-based CF worker (Phase S2) |

### Constants (top of file)
```python
MIN_RATINGS_PER_MOVIE = 5      # Filter sparse movies
S1_RATING_THRESHOLD = 2.5      # Phase S1 filtering cutoff
POSITIVE_RATING_THRESHOLD = 3.5 # Binary classification threshold
```

## Running the System

```bash
# Single-threaded (default)
python hw4.py -k 10 -t 0.8 -m 1

# Multi-threaded (8 workers)
python hw4.py -k 10 -t 0.8 -m 1 -w 8

# With reproducible results
python hw4.py -k 10 -t 0.8 -m 1 -s 42 -w 8

# Interactive mode
python hw4.py -i
```

**CLI Arguments:**
- `-k/--neighbors`: K value for KNN (default: 10)
- `-t/--train-ratio`: Training ratio 0-0.9 (default: 0.8)
- `-m/--model`: 1=Jaccard+Cosine, 2=Cosine+Cosine
- `-w/--workers`: Number of parallel workers (default: 1)
- `-s/--seed`: Random seed for reproducibility
- `-f/--file`: Path to ratings CSV

## Performance Benchmarks

### MovieLens 100K (~100K ratings)
| Workers | S1 Speed | S2 Speed | Total Time |
|---------|----------|----------|------------|
| 1       | 2,200 it/s | 2,400 it/s | ~9s |
| 8       | 12,000+ it/s | 13,000+ it/s | ~4s |

### MovieLens 1M (1M ratings)
| Workers | S1 Speed | S2 Speed | Total Time |
|---------|----------|----------|------------|
| 1       | 2,161 it/s | 1,573 it/s | 1:55 |
| 8       | 12,096 it/s | 9,108 it/s | 0:25 |

## Code Patterns

### Multiprocessing (initializer pattern, NOT partial)
```python
# CORRECT: Data loaded once per process via initializer
def _init_s1_worker(sim_matrices, train_idx, k):
    global _worker_data
    _worker_data = {'sim': sim_matrices, 'idx': train_idx, 'k': k}

with mp.Pool(workers, initializer=_init_s1_worker, 
             initargs=(sim_matrices, train_idx, k)) as pool:
    results = pool.imap_unordered(_predict_worker, tasks)

# WRONG: Pickling data with each task (extremely slow)
# partial_func = partial(predict, sim_matrices=..., train_idx=...)
```

### Vectorized Similarity (NOT loops)
```python
# Jaccard: matrix multiplication for intersection/union
intersection = binary_matrix.T @ binary_matrix
# Cosine: sklearn's optimized implementation
similarity = cosine_similarity(centered_matrix)
```

### Pre-indexed Training Data (NOT DataFrame filtering)
```python
# CORRECT: O(1) dict lookup
user_ratings = train_idx.by_user.get(user_id, pd.DataFrame())

# WRONG: O(n) filter on every prediction
user_ratings = train_df[train_df['userId'] == user_id]
```

### Prediction Collection (NOT pd.concat in loop)
```python
predictions = []  # Collect in list
for ...:
    predictions.append({...})
predictions_df = pd.DataFrame(predictions)  # Single DataFrame creation
```

### Type Hints & Dataclasses
All functions have type hints. Use dataclasses for structured data.

## Dependencies

```bash
pip install -r requirements.txt
```
- `numpy`, `pandas`: Data manipulation
- `scipy`: Sparse matrices
- `scikit-learn`: Cosine similarity, MAE
- `tqdm`: Progress bars

## Important Notes

- Similarity matrices are **numpy arrays** (not dicts) for O(1) lookup
- Index mappings (`movie_to_idx`, `user_to_idx`) translate IDs to array indices
- `TrainDataIndex` pre-groups data by user/movie for fast lookups
- Multiprocessing uses global `_worker_data` dict set by initializer
- Phase S1 keeps entries where prediction is `None` (lets S2 handle them)
- Division by zero is handled in precision/recall calculations
