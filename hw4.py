"""
Movie Recommendation System using Collaborative Filtering.

This system predicts user ratings using a two-phase approach:
- Phase S1 (Item-based): Filters candidates using movie-movie similarity
- Phase S2 (User-based): Predicts ratings using user-user similarity

Uses the MovieLens dataset and evaluates predictions via MAE, Precision, and Recall.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


# =============================================================================
# Constants
# =============================================================================

MIN_RATINGS_PER_MOVIE = 5
TEST_SPLIT_RATIO = 0.1
MAX_TRAIN_RATIO = 0.9
S1_RATING_THRESHOLD = 2.5
POSITIVE_RATING_THRESHOLD = 3.5


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

def run_two_phase_prediction(
    test_data: pd.DataFrame,
    train_data: pd.DataFrame,
    similarities: SimilarityMatrices,
    k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the two-phase prediction pipeline.
    
    Phase S1: Item-based filtering (drops predictions < 2.5)
    Phase S2: User-based final prediction
    
    Args:
        test_data: Test DataFrame.
        train_data: Training DataFrame.
        similarities: Precomputed similarity matrices.
        k: Number of neighbors.
        
    Returns:
        Tuple of (filtered_test_data, predictions_dataframe).
    """
    # Phase S1: Item-based filtering
    print("\nPhase S1: Item-based filtering...")
    s1_results = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc='Phase S1'):
        user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['rating']
        
        pred = predict_rating_item_based(user_id, movie_id, train_data, similarities, k)
        
        # Keep if prediction >= threshold or if we couldn't predict (let S2 handle it)
        if pred is None or pred >= S1_RATING_THRESHOLD:
            s1_results.append({
                'userId': user_id,
                'movieId': movie_id,
                'actual_rating': actual_rating,
                's1_prediction': pred
            })
    
    s1_df = pd.DataFrame(s1_results)
    print(f"  Kept {len(s1_df)} / {len(test_data)} entries after S1 filtering")
    
    # Phase S2: User-based prediction
    print("\nPhase S2: User-based prediction...")
    predictions = []
    
    for _, row in tqdm(s1_df.iterrows(), total=len(s1_df), desc='Phase S2'):
        user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['actual_rating']
        
        pred = predict_rating_user_based(user_id, movie_id, train_data, similarities, k)
        
        if pred is not None:
            predictions.append({
                'userId': user_id,
                'movieId': movie_id,
                'actual_rating': actual_rating,
                'predicted_rating': pred
            })
    
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
        default='ratings.csv',
        help='Path to ratings CSV file'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (prompts for parameters)'
    )
    
    return parser.parse_args()


def get_interactive_params() -> tuple[int, float, int]:
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
    
    return k, train_ratio, model


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Main entry point for the recommendation system."""
    args = parse_args()
    
    # Get parameters
    if args.interactive:
        k, train_ratio, model_choice = get_interactive_params()
    else:
        k = args.neighbors
        train_ratio = min(args.train_ratio, MAX_TRAIN_RATIO)
        model_choice = args.model
    
    print(f"\nConfiguration:")
    print(f"  K neighbors: {k}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Model: {model_choice} ({'Jaccard+Cosine' if model_choice == 1 else 'Cosine+Cosine'})")
    print(f"  Data file: {args.file}")
    
    # Load and preprocess data
    print("\nLoading data...")
    ratings = read_ratings(args.file)
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
    
    # Run prediction pipeline
    _, predictions_df = run_two_phase_prediction(
        test_data, train_data, similarities, k
    )
    
    # Evaluate
    metrics = evaluate_predictions(predictions_df)
    print_metrics(metrics)
    
    print("\n###### End of the program ######")


if __name__ == '__main__':
    main()
