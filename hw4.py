"""
Movie Recommendation System using Collaborative Filtering.

This system predicts user ratings using a two-phase approach:
- Phase S1 (Item-based): Filters candidates using movie-movie similarity
- Phase S2 (User-based): Predicts ratings using user-user similarity

Uses the MovieLens dataset and evaluates predictions via MAE, Precision, and Recall.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# Default number of workers (use CPU count - 1, minimum 1)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Global variables for multiprocessing workers (initialized once per process)
_worker_data: dict = {}


def _init_s1_worker(
    user_to_ratings: dict,
    movie_similarity: np.ndarray,
    movie_to_idx: dict,
    idx_to_movie: dict,
    k: int
) -> None:
    """Initialize global data for S1 workers (called once per process)."""
    _worker_data['user_to_ratings'] = user_to_ratings
    _worker_data['movie_similarity'] = movie_similarity
    _worker_data['movie_to_idx'] = movie_to_idx
    _worker_data['idx_to_movie'] = idx_to_movie
    _worker_data['k'] = k


def _init_s2_worker(
    movie_to_ratings: dict,
    user_similarity: np.ndarray,
    user_to_idx: dict,
    idx_to_user: dict,
    k: int
) -> None:
    """Initialize global data for S2 workers (called once per process)."""
    _worker_data['movie_to_ratings'] = movie_to_ratings
    _worker_data['user_similarity'] = user_similarity
    _worker_data['user_to_idx'] = user_to_idx
    _worker_data['idx_to_user'] = idx_to_user
    _worker_data['k'] = k


# =============================================================================
# Constants
# =============================================================================

MIN_RATINGS_PER_MOVIE = 5
TEST_SPLIT_RATIO = 0.1
MAX_TRAIN_RATIO = 0.9
S1_RATING_THRESHOLD = 2.5
POSITIVE_RATING_THRESHOLD = 3.5

# Dataset configurations
DATASETS = {
    '100k': {
        'name': 'MovieLens 100K',
        'url': 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'zip_name': 'ml-latest-small.zip',
        'extract_dir': 'ml-latest-small',
        'ratings_file': 'ratings.csv',
        'format': 'csv',  # Standard CSV with header
        'size': '~100K ratings, ~1GB RAM',
    },
    '1m': {
        'name': 'MovieLens 1M',
        'url': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'zip_name': 'ml-1m.zip',
        'extract_dir': 'ml-1m',
        'ratings_file': 'ratings.dat',
        'format': 'dat',  # :: delimited
        'size': '~1M ratings, ~2GB RAM',
    },
    '10m': {
        'name': 'MovieLens 10M',
        'url': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'zip_name': 'ml-10m.zip',
        'extract_dir': 'ml-10M100K',
        'ratings_file': 'ratings.dat',
        'format': 'dat',  # :: delimited
        'size': '~10M ratings, ~40GB RAM',
    },
    '25m': {
        'name': 'MovieLens 25M',
        'url': 'https://files.grouplens.org/datasets/movielens/ml-25m.zip',
        'zip_name': 'ml-25m.zip',
        'extract_dir': 'ml-25m',
        'ratings_file': 'ratings.csv',
        'format': 'csv',
        'size': '~25M ratings, ~200GB RAM',
    },
    'local': {
        'name': 'Local file (ratings.csv)',
        'size': 'Use existing local file',
    },
}

DATA_DIR = Path('data')


# =============================================================================
# Dataset Download and Management
# =============================================================================

class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> None:
    """Download a file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def get_dataset_path(dataset_key: str) -> Path:
    """Get the path to a dataset's ratings file (always CSV)."""
    if dataset_key == 'local':
        return Path('ratings.csv')
    
    config = DATASETS[dataset_key]
    base_path = DATA_DIR / config['extract_dir'] / config['ratings_file']
    
    # For .dat format datasets, we use the converted .csv file
    if config['format'] == 'dat':
        return base_path.with_suffix('.csv')
    return base_path


def is_dataset_available(dataset_key: str) -> bool:
    """Check if a dataset is already downloaded and converted."""
    if dataset_key == 'local':
        return Path('ratings.csv').exists()
    return get_dataset_path(dataset_key).exists()


def download_dataset(dataset_key: str) -> Path:
    """
    Download and extract a MovieLens dataset if not already present.
    
    Args:
        dataset_key: Key from DATASETS dict (e.g., '100k', '1m', '10m', '25m')
        
    Returns:
        Path to the ratings CSV file.
    """
    if dataset_key == 'local':
        path = Path('ratings.csv')
        if not path.exists():
            raise FileNotFoundError("Local ratings.csv not found")
        return path
    
    config = DATASETS[dataset_key]
    csv_path = get_dataset_path(dataset_key)  # Always returns CSV path
    
    # Check if already downloaded and converted
    if csv_path.exists():
        print(f"  Dataset already available: {csv_path}")
        return csv_path
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    zip_path = DATA_DIR / config['zip_name']
    
    # Download if zip doesn't exist
    if not zip_path.exists():
        print(f"  Downloading {config['name']}...")
        download_file(config['url'], zip_path)
    
    # Extract
    print(f"  Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    
    # Clean up zip file
    zip_path.unlink()
    
    # Convert .dat format to CSV if needed
    if config['format'] == 'dat':
        dat_path = DATA_DIR / config['extract_dir'] / config['ratings_file']
        print(f"  Converting {dat_path.name} to CSV format...")
        convert_dat_to_csv(dat_path, csv_path)
    
    return csv_path


def convert_dat_to_csv(dat_path: Path, csv_path: Path) -> None:
    """Convert MovieLens .dat format (:: delimiter) to CSV."""
    # Read with :: delimiter, no header
    df = pd.read_csv(
        dat_path, 
        sep='::', 
        engine='python',
        names=['userId', 'movieId', 'rating', 'timestamp'],
        dtype={'userId': str, 'movieId': str, 'rating': float}
    )
    # Save as CSV
    df.to_csv(csv_path, index=False)


def select_dataset_interactive() -> str:
    """Interactive dataset selection menu."""
    print("\n" + "=" * 50)
    print("DATASET SELECTION")
    print("=" * 50)
    
    keys = list(DATASETS.keys())
    for i, key in enumerate(keys, 1):
        config = DATASETS[key]
        status = "✓ available" if is_dataset_available(key) else "↓ will download"
        print(f"  {i}. {config['name']:25} ({config['size']}) [{status}]")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(keys)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                return keys[idx]
            print(f"Please enter a number between 1 and {len(keys)}")
        except ValueError:
            print("Please enter a valid number")


def prepare_dataset(dataset_key: str | None = None, file_path: str | None = None) -> Path:
    """
    Prepare the dataset for use, downloading if necessary.
    
    Args:
        dataset_key: Key from DATASETS dict, or None for interactive selection
        file_path: Direct path to ratings file (overrides dataset_key)
        
    Returns:
        Path to the ratings file ready for use.
    """
    # If direct file path provided, use it
    if file_path and Path(file_path).exists():
        return Path(file_path)
    
    # Interactive selection if no dataset specified
    if dataset_key is None:
        dataset_key = select_dataset_interactive()
    
    print(f"\nPreparing dataset: {DATASETS[dataset_key]['name']}")
    return download_dataset(dataset_key)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SimilarityMatrices:
    """Container for precomputed similarity matrices and index mappings."""
    movie_similarity: np.ndarray
    user_similarity: np.ndarray
    movie_to_idx: dict[str, int]
    idx_to_movie: dict[int, str]
    user_to_idx: dict[str, int]
    idx_to_user: dict[int, str]


@dataclass
class TrainDataIndex:
    """Pre-indexed training data for fast lookups."""
    user_to_ratings: dict[str, pd.DataFrame]  # user_id -> their ratings
    movie_to_ratings: dict[str, pd.DataFrame]  # movie_id -> ratings for that movie


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    precision: float
    recall: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


# =============================================================================
# Data Loading
# =============================================================================

def read_ratings(filename: str) -> pd.DataFrame:
    """
    Load ratings from a CSV file.
    
    Args:
        filename: Path to the ratings CSV file.
        
    Returns:
        DataFrame with columns: userId, movieId, rating
    """
    ratings = pd.read_csv(
        filename,
        usecols=['userId', 'movieId', 'rating'],
        dtype={'userId': str, 'movieId': str, 'rating': float}
    )
    return ratings.dropna()


def filter_sparse_movies(ratings: pd.DataFrame, min_ratings: int = MIN_RATINGS_PER_MOVIE) -> pd.DataFrame:
    """
    Remove movies with fewer than min_ratings ratings.
    
    Args:
        ratings: DataFrame with rating data.
        min_ratings: Minimum number of ratings required for a movie.
        
    Returns:
        Filtered DataFrame.
    """
    movie_counts = ratings.groupby('movieId').size()
    valid_movies = movie_counts[movie_counts >= min_ratings].index
    return ratings[ratings['movieId'].isin(valid_movies)].copy()


def train_test_split(
    ratings: pd.DataFrame, 
    test_ratio: float = TEST_SPLIT_RATIO,
    train_ratio: float = 0.8,
    random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings into training and test sets.
    
    Args:
        ratings: DataFrame with rating data.
        test_ratio: Fraction of data to use for testing.
        train_ratio: Fraction of remaining data to use for training.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_data, test_data).
    """
    # Sample test data
    test_data = ratings.sample(frac=test_ratio, random_state=random_state)
    
    # Sample training data from remaining
    remaining = ratings.drop(test_data.index)
    train_count = int(train_ratio * len(ratings))
    train_data = remaining.sample(n=min(train_count, len(remaining)), random_state=random_state)
    
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)


# =============================================================================
# Similarity Computation (Vectorized)
# =============================================================================

def compute_jaccard_similarity_matrix(user_item_matrix: csr_matrix) -> np.ndarray:
    """
    Compute Jaccard similarity matrix for items using vectorized operations.
    
    Args:
        user_item_matrix: Sparse matrix of shape (n_users, n_items) with binary values.
        
    Returns:
        Similarity matrix of shape (n_items, n_items).
    """
    # Convert to binary (rated/not rated)
    binary_matrix = (user_item_matrix > 0).astype(float)
    
    # Intersection: A^T @ A gives co-occurrence counts
    intersection = binary_matrix.T @ binary_matrix
    intersection = intersection.toarray()
    
    # Union: |A| + |B| - |A ∩ B|
    item_counts = np.array(binary_matrix.sum(axis=0)).flatten()
    union = item_counts[:, np.newaxis] + item_counts[np.newaxis, :] - intersection
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        jaccard = np.where(union > 0, intersection / union, 0)
    
    # Zero out diagonal
    np.fill_diagonal(jaccard, 0)
    
    return jaccard


def compute_adjusted_cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Compute adjusted cosine similarity using sklearn's optimized implementation.
    
    Args:
        matrix: Mean-centered rating matrix.
        
    Returns:
        Similarity matrix.
    """
    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)
    return similarity


def build_similarity_matrices(
    train_data: pd.DataFrame, 
    model_choice: Literal[1, 2]
) -> SimilarityMatrices:
    """
    Build movie and user similarity matrices using vectorized operations.
    
    Args:
        train_data: Training DataFrame with userId, movieId, rating columns.
        model_choice: 1 for Jaccard (movies) + Cosine (users), 2 for Cosine (both).
        
    Returns:
        SimilarityMatrices containing precomputed similarities and index mappings.
    """
    # Create index mappings
    unique_users = train_data['userId'].unique()
    unique_movies = train_data['movieId'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
    idx_to_movie = {idx: movie for movie, idx in movie_to_idx.items()}
    
    # Build user-item matrix
    n_users, n_movies = len(unique_users), len(unique_movies)
    
    user_indices = train_data['userId'].map(user_to_idx).values
    movie_indices = train_data['movieId'].map(movie_to_idx).values
    ratings = train_data['rating'].values
    
    user_item_matrix = csr_matrix(
        (ratings, (user_indices, movie_indices)),
        shape=(n_users, n_movies)
    )
    
    # Convert to dense for similarity computation
    dense_matrix = user_item_matrix.toarray()
    
    # Mean-center per user (for adjusted cosine)
    user_means = np.nanmean(np.where(dense_matrix == 0, np.nan, dense_matrix), axis=1, keepdims=True)
    user_means = np.nan_to_num(user_means, nan=0)
    centered_matrix = np.where(dense_matrix == 0, 0, dense_matrix - user_means)
    
    # Compute movie similarity
    print("\nComputing movie similarities...")
    if model_choice == 1:
        movie_similarity = compute_jaccard_similarity_matrix(user_item_matrix)
    else:
        # Transpose: rows = movies, cols = users
        movie_similarity = compute_adjusted_cosine_similarity(centered_matrix.T)
    
    # Compute user similarity (always adjusted cosine)
    print("Computing user similarities...")
    user_similarity = compute_adjusted_cosine_similarity(centered_matrix)
    
    return SimilarityMatrices(
        movie_similarity=movie_similarity,
        user_similarity=user_similarity,
        movie_to_idx=movie_to_idx,
        idx_to_movie=idx_to_movie,
        user_to_idx=user_to_idx,
        idx_to_user=idx_to_user
    )


def build_train_data_index(train_data: pd.DataFrame) -> TrainDataIndex:
    """
    Pre-index training data for fast lookups during prediction.
    
    Args:
        train_data: Training DataFrame.
        
    Returns:
        TrainDataIndex with pre-grouped data.
    """
    print("Building training data index...")
    user_to_ratings = {user: group for user, group in train_data.groupby('userId')}
    movie_to_ratings = {movie: group for movie, group in train_data.groupby('movieId')}
    return TrainDataIndex(user_to_ratings=user_to_ratings, movie_to_ratings=movie_to_ratings)


# =============================================================================
# Prediction Functions
# =============================================================================

def predict_rating_item_based(
    user_id: str,
    movie_id: str,
    train_data: pd.DataFrame,
    similarities: SimilarityMatrices,
    k: int
) -> float | None:
    """
    Predict rating using item-based collaborative filtering.
    
    Args:
        user_id: Target user ID.
        movie_id: Target movie ID.
        train_data: Training data DataFrame.
        similarities: Precomputed similarity matrices.
        k: Number of neighbors to use.
        
    Returns:
        Predicted rating or None if prediction not possible.
    """
    if movie_id not in similarities.movie_to_idx:
        return None
    
    movie_idx = similarities.movie_to_idx[movie_id]
    
    # Get user's rated movies from training data
    user_ratings = train_data[train_data['userId'] == user_id]
    if user_ratings.empty:
        return None
    
    # Get similarities for rated movies
    rated_movie_ids = user_ratings['movieId'].values
    rated_movie_indices = [
        similarities.movie_to_idx[m] 
        for m in rated_movie_ids 
        if m in similarities.movie_to_idx
    ]
    
    if not rated_movie_indices:
        return None
    
    # Get similarity scores
    sims = similarities.movie_similarity[movie_idx, rated_movie_indices]
    ratings = user_ratings[user_ratings['movieId'].isin(
        [similarities.idx_to_movie[i] for i in rated_movie_indices]
    )]['rating'].values
    
    # Filter positive similarities and get top-k
    positive_mask = sims > 0
    if not positive_mask.any():
        return None
    
    sims = sims[positive_mask]
    ratings = ratings[positive_mask]
    
    # Get top-k
    if len(sims) > k:
        top_k_indices = np.argsort(sims)[-k:]
        sims = sims[top_k_indices]
        ratings = ratings[top_k_indices]
    
    # Weighted average
    return float(np.dot(ratings, sims) / sims.sum())


def predict_rating_user_based(
    user_id: str,
    movie_id: str,
    train_data: pd.DataFrame,
    similarities: SimilarityMatrices,
    k: int
) -> float | None:
    """
    Predict rating using user-based collaborative filtering.
    
    Args:
        user_id: Target user ID.
        movie_id: Target movie ID.
        train_data: Training data DataFrame.
        similarities: Precomputed similarity matrices.
        k: Number of neighbors to use.
        
    Returns:
        Predicted rating or None if prediction not possible.
    """
    if user_id not in similarities.user_to_idx:
        return None
    
    user_idx = similarities.user_to_idx[user_id]
    
    # Get users who rated this movie
    movie_ratings = train_data[train_data['movieId'] == movie_id]
    if movie_ratings.empty:
        return None
    
    # Get similarities for those users
    rater_ids = movie_ratings['userId'].values
    rater_indices = [
        similarities.user_to_idx[u] 
        for u in rater_ids 
        if u in similarities.user_to_idx
    ]
    
    if not rater_indices:
        return None
    
    # Get similarity scores
    sims = similarities.user_similarity[user_idx, rater_indices]
    ratings = movie_ratings[movie_ratings['userId'].isin(
        [similarities.idx_to_user[i] for i in rater_indices]
    )]['rating'].values
    
    # Filter positive similarities and get top-k
    positive_mask = sims > 0
    if not positive_mask.any():
        return None
    
    sims = sims[positive_mask]
    ratings = ratings[positive_mask]
    
    # Get top-k
    if len(sims) > k:
        top_k_indices = np.argsort(sims)[-k:]
        sims = sims[top_k_indices]
        ratings = ratings[top_k_indices]
    
    # Weighted average
    return float(np.dot(ratings, sims) / sims.sum())


# =============================================================================
# Two-Phase Prediction Pipeline
# =============================================================================

def _predict_item_based_worker(
    row_data: tuple[str, str, float],
    user_to_ratings: dict[str, pd.DataFrame] | None = None,
    movie_similarity: np.ndarray | None = None,
    movie_to_idx: dict[str, int] | None = None,
    idx_to_movie: dict[int, str] | None = None,
    k: int | None = None
) -> dict | None:
    """Worker function for parallel item-based prediction."""
    # Use global data if not provided (multiprocessing mode)
    if user_to_ratings is None:
        user_to_ratings = _worker_data['user_to_ratings']
        movie_similarity = _worker_data['movie_similarity']
        movie_to_idx = _worker_data['movie_to_idx']
        idx_to_movie = _worker_data['idx_to_movie']
        k = _worker_data['k']
    
    user_id, movie_id, actual_rating = row_data
    
    if movie_id not in movie_to_idx:
        return None
    
    movie_idx = movie_to_idx[movie_id]
    
    # Use pre-indexed lookup instead of DataFrame filtering
    if user_id not in user_to_ratings:
        return None
    user_ratings = user_to_ratings[user_id]
    
    if user_ratings.empty:
        return None
    
    rated_movie_ids = user_ratings['movieId'].values
    rated_movie_indices = [movie_to_idx[m] for m in rated_movie_ids if m in movie_to_idx]
    
    if not rated_movie_indices:
        return None
    
    sims = movie_similarity[movie_idx, rated_movie_indices]
    ratings = user_ratings[user_ratings['movieId'].isin(
        [idx_to_movie[i] for i in rated_movie_indices]
    )]['rating'].values
    
    positive_mask = sims > 0
    if not positive_mask.any():
        return None
    
    sims = sims[positive_mask]
    ratings = ratings[positive_mask]
    
    if len(sims) > k:
        top_k_indices = np.argsort(sims)[-k:]
        sims = sims[top_k_indices]
        ratings = ratings[top_k_indices]
    
    pred = float(np.dot(ratings, sims) / sims.sum())
    
    # Return result only if passes S1 threshold
    if pred >= S1_RATING_THRESHOLD:
        return {
            'userId': user_id,
            'movieId': movie_id,
            'actual_rating': actual_rating,
            's1_prediction': pred
        }
    return None


def _predict_user_based_worker(
    row_data: tuple[str, str, float],
    movie_to_ratings: dict[str, pd.DataFrame] | None = None,
    user_similarity: np.ndarray | None = None,
    user_to_idx: dict[str, int] | None = None,
    idx_to_user: dict[int, str] | None = None,
    k: int | None = None
) -> dict | None:
    """Worker function for parallel user-based prediction."""
    # Use global data if not provided (multiprocessing mode)
    if movie_to_ratings is None:
        movie_to_ratings = _worker_data['movie_to_ratings']
        user_similarity = _worker_data['user_similarity']
        user_to_idx = _worker_data['user_to_idx']
        idx_to_user = _worker_data['idx_to_user']
        k = _worker_data['k']
    
    user_id, movie_id, actual_rating = row_data
    
    if user_id not in user_to_idx:
        return None
    
    user_idx = user_to_idx[user_id]
    
    # Use pre-indexed lookup instead of DataFrame filtering
    if movie_id not in movie_to_ratings:
        return None
    movie_ratings = movie_to_ratings[movie_id]
    
    if movie_ratings.empty:
        return None
    
    rater_ids = movie_ratings['userId'].values
    rater_indices = [user_to_idx[u] for u in rater_ids if u in user_to_idx]
    
    if not rater_indices:
        return None
    
    sims = user_similarity[user_idx, rater_indices]
    ratings = movie_ratings[movie_ratings['userId'].isin(
        [idx_to_user[i] for i in rater_indices]
    )]['rating'].values
    
    positive_mask = sims > 0
    if not positive_mask.any():
        return None
    
    sims = sims[positive_mask]
    ratings = ratings[positive_mask]
    
    if len(sims) > k:
        top_k_indices = np.argsort(sims)[-k:]
        sims = sims[top_k_indices]
        ratings = ratings[top_k_indices]
    
    pred = float(np.dot(ratings, sims) / sims.sum())
    
    return {
        'userId': user_id,
        'movieId': movie_id,
        'actual_rating': actual_rating,
        'predicted_rating': pred
    }


def run_two_phase_prediction(
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    similarities: SimilarityMatrices,
    train_index: TrainDataIndex,
    k: int,
    n_workers: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the two-phase prediction pipeline.
    
    Phase S1: Item-based filtering (drops predictions < 2.5)
    Phase S2: User-based final prediction
    
    Args:
        test_data: Test DataFrame.
        train_data: Training DataFrame.
        similarities: Precomputed similarity matrices.
        train_index: Pre-indexed training data for fast lookups.
        k: Number of neighbors.
        n_workers: Number of parallel workers (1 = single-threaded).
        
    Returns:
        Tuple of (filtered_test_data, predictions_dataframe).
    """
    # Prepare data as list of tuples for workers
    test_rows = list(zip(
        test_data['userId'].values,
        test_data['movieId'].values,
        test_data['rating'].values
    ))
    
    # Phase S1: Item-based filtering
    print(f"\nPhase S1: Item-based filtering... (workers: {n_workers})")
    
    if n_workers > 1:
        # Use initializer to share data across workers (avoids pickling per task)
        with mp.Pool(
            n_workers,
            initializer=_init_s1_worker,
            initargs=(
                train_index.user_to_ratings,
                similarities.movie_similarity,
                similarities.movie_to_idx,
                similarities.idx_to_movie,
                k
            )
        ) as pool:
            s1_results = list(tqdm(
                pool.imap(_predict_item_based_worker, test_rows, chunksize=500),
                total=len(test_rows),
                desc='Phase S1'
            ))
    else:
        # Single-threaded fallback
        s1_results = []
        for row in tqdm(test_rows, desc='Phase S1'):
            result = _predict_item_based_worker(
                row, 
                train_index.user_to_ratings,
                similarities.movie_similarity,
                similarities.movie_to_idx,
                similarities.idx_to_movie,
                k
            )
            # Also include items we couldn't predict (let S2 handle them)
            if result is None:
                user_id, movie_id, actual_rating = row
                if movie_id in similarities.movie_to_idx:
                    result = {
                        'userId': user_id,
                        'movieId': movie_id,
                        'actual_rating': actual_rating,
                        's1_prediction': None
                    }
            s1_results.append(result)
    
    # Filter None results and handle unpredicted items for parallel case
    if n_workers > 1:
        filtered_results = []
        for i, result in enumerate(s1_results):
            if result is not None:
                filtered_results.append(result)
            else:
                # Check if movie exists - if so, let S2 try
                user_id, movie_id, actual_rating = test_rows[i]
                if movie_id in similarities.movie_to_idx:
                    # Item-based couldn't predict, but movie exists - let S2 try
                    filtered_results.append({
                        'userId': user_id,
                        'movieId': movie_id,
                        'actual_rating': actual_rating,
                        's1_prediction': None
                    })
        s1_results = filtered_results
    else:
        s1_results = [r for r in s1_results if r is not None]
    
    s1_df = pd.DataFrame(s1_results)
    print(f"  Kept {len(s1_df)} / {len(test_data)} entries after S1 filtering")
    
    if s1_df.empty:
        return s1_df, pd.DataFrame()
    
    # Phase S2: User-based prediction
    print(f"\nPhase S2: User-based prediction... (workers: {n_workers})")
    
    s2_rows = list(zip(
        s1_df['userId'].values,
        s1_df['movieId'].values,
        s1_df['actual_rating'].values
    ))
    
    if n_workers > 1:
        # Use initializer to share data across workers (avoids pickling per task)
        with mp.Pool(
            n_workers,
            initializer=_init_s2_worker,
            initargs=(
                train_index.movie_to_ratings,
                similarities.user_similarity,
                similarities.user_to_idx,
                similarities.idx_to_user,
                k
            )
        ) as pool:
            predictions = list(tqdm(
                pool.imap(_predict_user_based_worker, s2_rows, chunksize=500),
                total=len(s2_rows),
                desc='Phase S2'
            ))
    else:
        predictions = []
        for row in tqdm(s2_rows, desc='Phase S2'):
            result = _predict_user_based_worker(
                row, 
                train_index.movie_to_ratings,
                similarities.user_similarity,
                similarities.user_to_idx,
                similarities.idx_to_user,
                k
            )
            predictions.append(result)
    
    # Filter None results
    predictions = [p for p in predictions if p is not None]
    
    predictions_df = pd.DataFrame(predictions)
    print(f"  Generated {len(predictions_df)} predictions")
    
    return s1_df, predictions_df


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_predictions(predictions_df: pd.DataFrame) -> EvaluationMetrics:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        predictions_df: DataFrame with 'actual_rating' and 'predicted_rating' columns.
        
    Returns:
        EvaluationMetrics dataclass with all metrics.
    """
    if predictions_df.empty:
        return EvaluationMetrics(
            mae=float('inf'),
            precision=0.0,
            recall=0.0,
            true_positives=0,
            true_negatives=0,
            false_positives=0,
            false_negatives=0
        )
    
    actual = predictions_df['actual_rating'].values
    predicted = predictions_df['predicted_rating'].values
    
    # MAE
    mae = mean_absolute_error(actual, predicted)
    
    # Binary classification metrics
    actual_positive = actual > POSITIVE_RATING_THRESHOLD
    predicted_positive = predicted > POSITIVE_RATING_THRESHOLD
    
    tp = int(np.sum(actual_positive & predicted_positive))
    tn = int(np.sum(~actual_positive & ~predicted_positive))
    fp = int(np.sum(~actual_positive & predicted_positive))
    fn = int(np.sum(actual_positive & ~predicted_positive))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return EvaluationMetrics(
        mae=mae,
        precision=precision,
        recall=recall,
        true_positives=tp,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn
    )


def print_metrics(metrics: EvaluationMetrics) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean Absolute Error: {metrics.mae:.4f}")
    print(f"Precision:           {metrics.precision:.4f}")
    print(f"Recall:              {metrics.recall:.4f}")
    print("-" * 50)
    print(f"True Positives:  {metrics.true_positives}")
    print(f"True Negatives:  {metrics.true_negatives}")
    print(f"False Positives: {metrics.false_positives}")
    print(f"False Negatives: {metrics.false_negatives}")
    print("=" * 50)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Movie Recommendation System using Collaborative Filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-d', '--dataset',
        type=str,
        choices=list(DATASETS.keys()),
        default=None,
        help='Dataset to use (will download if not present)'
    )
    parser.add_argument(
        '-k', '--neighbors',
        type=int,
        default=10,
        help='Number of neighbors for KNN prediction'
    )
    parser.add_argument(
        '-t', '--train-ratio',
        type=float,
        default=0.8,
        help='Fraction of data to use for training (max 0.9)'
    )
    parser.add_argument(
        '-m', '--model',
        type=int,
        choices=[1, 2],
        default=1,
        help='Model choice: 1=Jaccard+Cosine, 2=Cosine+Cosine'
    )
    parser.add_argument(
        '-f', '--file',
        type=str,
        default=None,
        help='Direct path to ratings CSV file (overrides -d/--dataset)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (prompts for parameters)'
    )
    
    return parser.parse_args()


def get_interactive_params() -> tuple[int, float, int, int]:
    """Get parameters interactively from user input."""
    while True:
        try:
            k = int(input('Enter the K value (neighbors): '))
            if k >= 1:
                break
            print('K value must be greater than 0')
        except ValueError:
            print('Please enter a valid integer')
    
    while True:
        try:
            train_ratio = float(input('Training data ratio (e.g., 0.8 for 80%): '))
            if 0 < train_ratio <= MAX_TRAIN_RATIO:
                break
            print(f'Ratio must be between 0 and {MAX_TRAIN_RATIO}')
        except ValueError:
            print('Please enter a valid number')
    
    while True:
        try:
            model = int(input('Model (1=Jaccard+Cosine, 2=Cosine+Cosine): '))
            if model in [1, 2]:
                break
            print('Model choice must be 1 or 2')
        except ValueError:
            print('Please enter a valid integer')
    
    max_workers = os.cpu_count() or 4
    while True:
        try:
            workers = input(f'Number of workers (1-{max_workers}, default={DEFAULT_WORKERS}): ').strip()
            if workers == '':
                workers = DEFAULT_WORKERS
                break
            workers = int(workers)
            if 1 <= workers <= max_workers:
                break
            print(f'Workers must be between 1 and {max_workers}')
        except ValueError:
            print('Please enter a valid integer')
    
    return k, train_ratio, model, workers


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point for the recommendation system."""
    args = parse_args()
    
    # Dataset selection / preparation
    if args.interactive:
        # Interactive mode: show dataset menu first
        data_file = prepare_dataset(dataset_key=None, file_path=args.file)
        k, train_ratio, model_choice, n_workers = get_interactive_params()
    else:
        # CLI mode
        k = args.neighbors
        train_ratio = min(args.train_ratio, MAX_TRAIN_RATIO)
        model_choice = args.model
        n_workers = max(1, args.workers)
        
        # Determine dataset
        if args.file:
            data_file = prepare_dataset(file_path=args.file)
        elif args.dataset:
            data_file = prepare_dataset(dataset_key=args.dataset)
        else:
            # Default: interactive dataset selection
            data_file = prepare_dataset(dataset_key=None)
    
    print(f"\nConfiguration:")
    print(f"  K neighbors: {k}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Model: {model_choice} ({'Jaccard+Cosine' if model_choice == 1 else 'Cosine+Cosine'})")
    print(f"  Data file: {data_file}")
    print(f"  Workers: {n_workers}")
    
    # Load and preprocess data
    print("\nLoading data...")
    ratings = read_ratings(str(data_file))
    print(f"  Loaded {len(ratings)} ratings")
    
    ratings = filter_sparse_movies(ratings)
    print(f"  After filtering sparse movies: {len(ratings)} ratings")
    
    # Split data
    train_data, test_data = train_test_split(
        ratings, 
        train_ratio=train_ratio,
        random_state=args.seed
    )
    print(f"  Training set: {len(train_data)} ratings")
    print(f"  Test set: {len(test_data)} ratings")
    
    # Build similarity matrices
    similarities = build_similarity_matrices(train_data, model_choice)
    
    # Build training data index for fast lookups
    train_index = build_train_data_index(train_data)
    
    # Run prediction pipeline
    _, predictions_df = run_two_phase_prediction(
        test_data, train_data, similarities, train_index, k, n_workers
    )
    
    # Evaluate
    metrics = evaluate_predictions(predictions_df)
    print_metrics(metrics)
    
    print("\n###### End of the program ######")


if __name__ == '__main__':
    main()
