# Copilot Instructions for Movie Recommendation System

## Project Overview

**Collaborative filtering movie recommendation system** using a two-phase approach:
- **Phase S1 (Item-based)**: Filters candidates using movie-movie similarity (threshold: 2.5)
- **Phase S2 (User-based)**: Predicts final ratings using user-user similarity

Uses MovieLens dataset (~100K ratings) with MAE, Precision, and Recall evaluation.

## Architecture

### Data Flow
```
ratings.csv → read_ratings() → filter_sparse_movies() → train_test_split()
           → build_similarity_matrices() → run_two_phase_prediction() → evaluate_predictions()
```

### Key Components in `hw4.py`

| Component | Purpose |
|-----------|---------|
| `SimilarityMatrices` | Dataclass holding precomputed numpy arrays + index mappings |
| `EvaluationMetrics` | Dataclass for MAE, precision, recall, confusion matrix |
| `build_similarity_matrices()` | Vectorized similarity computation using scipy/sklearn |
| `predict_rating_item_based()` | Item-based CF with top-K neighbor selection |
| `predict_rating_user_based()` | User-based CF with top-K neighbor selection |

### Constants (top of file)
```python
MIN_RATINGS_PER_MOVIE = 5      # Filter sparse movies
S1_RATING_THRESHOLD = 2.5      # Phase S1 filtering cutoff
POSITIVE_RATING_THRESHOLD = 3.5 # Binary classification threshold
```

## Running the System

```bash
# CLI mode (default)
python hw4.py -k 10 -t 0.8 -m 1

# Interactive mode (prompts for input)
python hw4.py -i

# With reproducible results
python hw4.py -k 10 -t 0.8 -m 1 -s 42
```

**CLI Arguments:**
- `-k/--neighbors`: K value for KNN (default: 10)
- `-t/--train-ratio`: Training ratio 0-0.9 (default: 0.8)
- `-m/--model`: 1=Jaccard+Cosine, 2=Cosine+Cosine
- `-s/--seed`: Random seed for reproducibility
- `-f/--file`: Path to ratings CSV

## Code Patterns

### Vectorized Similarity (NOT loops)
```python
# Jaccard: matrix multiplication for intersection/union
intersection = binary_matrix.T @ binary_matrix
# Cosine: sklearn's optimized implementation
similarity = cosine_similarity(centered_matrix)
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
- Phase S1 keeps entries where prediction is `None` (lets S2 handle them)
- Division by zero is handled in precision/recall calculations
