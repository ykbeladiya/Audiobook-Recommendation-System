import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sparse
from typing import Tuple, Dict
import os
import json

def load_data(data_dir: str = 'data/raw') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load audiobooks and user interactions data."""
    audiobooks = pd.read_csv(os.path.join(data_dir, 'audiobooks.csv'))
    interactions = pd.read_csv(os.path.join(data_dir, 'user_interactions.csv'))
    return audiobooks, interactions

def create_user_item_matrix(interactions: pd.DataFrame) -> sparse.csr_matrix:
    """
    Create a sparse user-item interaction matrix.
    Matrix values are weighted by both progress and rating:
    - If rating exists: rating * (progress/100)
    - If no rating: progress/100
    """
    # Create user and item mappings
    user_ids = interactions['user_id'].unique()
    book_ids = interactions['book_id'].unique()
    
    user_to_idx = {int(uid): idx for idx, uid in enumerate(user_ids)}
    book_to_idx = {int(bid): idx for idx, bid in enumerate(book_ids)}
    
    # Save mappings for later use
    mappings = {
        'user_to_idx': {str(k): int(v) for k, v in user_to_idx.items()},
        'book_to_idx': {str(k): int(v) for k, v in book_to_idx.items()},
        'idx_to_user': {str(v): int(k) for k, v in user_to_idx.items()},
        'idx_to_book': {str(v): int(k) for k, v in book_to_idx.items()}
    }
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/id_mappings.json', 'w') as f:
        json.dump(mappings, f)
    
    # Calculate interaction values
    rows, cols, values = [], [], []
    for _, row in interactions.iterrows():
        user_idx = user_to_idx[int(row['user_id'])]
        book_idx = book_to_idx[int(row['book_id'])]
        progress_weight = row['progress'] / 100.0
        
        # Calculate interaction value
        if pd.isna(row['rating']):
            value = progress_weight
        else:
            value = row['rating'] * progress_weight
        
        rows.append(user_idx)
        cols.append(book_idx)
        values.append(value)
    
    # Create sparse matrix
    matrix = sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(user_ids), len(book_ids))
    )
    
    return matrix

def create_content_features(audiobooks: pd.DataFrame) -> Tuple[sparse.csr_matrix, Dict]:
    """
    Create content-based features using TF-IDF on descriptions and tags.
    Also includes genre as one-hot encoded features.
    """
    # Combine description and tags for text features
    audiobooks['text_features'] = audiobooks['description'] + ' ' + audiobooks['tags'].fillna('')
    
    # Create TF-IDF features from text
    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    text_features = tfidf.fit_transform(audiobooks['text_features'])
    
    # Create one-hot encoding for genres
    genres = pd.get_dummies(audiobooks['genre'], prefix='genre')
    genre_features = sparse.csr_matrix(genres.values)
    
    # Normalize duration to 0-1 range
    scaler = MinMaxScaler()
    duration_normalized = scaler.fit_transform(audiobooks[['duration']])
    duration_features = sparse.csr_matrix(duration_normalized)
    
    # Combine all features
    content_features = sparse.hstack([
        text_features,
        genre_features,
        duration_features
    ]).tocsr()
    
    # Save feature names for later use
    feature_names = {
        'tfidf': list(tfidf.get_feature_names_out()),
        'genres': list(genres.columns),
        'duration': ['duration']
    }
    
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    return content_features

def main():
    """Preprocess data and save results."""
    print("Loading data...")
    audiobooks, interactions = load_data()
    
    print("Creating user-item interaction matrix...")
    interaction_matrix = create_user_item_matrix(interactions)
    print(f"Created matrix with shape: {interaction_matrix.shape}")
    
    print("Creating content features...")
    content_features = create_content_features(audiobooks)
    print(f"Created features with shape: {content_features.shape}")
    
    # Save processed matrices
    print("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    sparse.save_npz('data/processed/user_item_matrix.npz', interaction_matrix)
    sparse.save_npz('data/processed/content_features.npz', content_features)
    
    print("Done! Processed data saved to data/processed/")

if __name__ == "__main__":
    main() 