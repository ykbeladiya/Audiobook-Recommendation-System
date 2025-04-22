import numpy as np
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import List, Dict, Tuple
import pandas as pd

class UserUserCollaborativeFiltering:
    def __init__(self, data_dir: str = 'data/processed'):
        """Initialize the recommender system."""
        # Load the user-item interaction matrix
        self.user_item_matrix = sparse.load_npz(
            os.path.join(data_dir, 'user_item_matrix.npz')
        )
        
        # Load ID mappings
        with open(os.path.join(data_dir, 'id_mappings.json'), 'r') as f:
            self.id_mappings = json.load(f)
        
        # Calculate user similarity matrix
        self.user_similarity = self._compute_user_similarity()
        
        # Load audiobooks data for additional information
        self.audiobooks = pd.read_csv('data/raw/audiobooks.csv')
    
    def _compute_user_similarity(self) -> np.ndarray:
        """Compute user-user similarity matrix using cosine similarity."""
        return cosine_similarity(self.user_item_matrix)
    
    def _get_user_idx(self, user_id: int) -> int:
        """Convert user ID to matrix index."""
        return self.id_mappings['user_to_idx'][str(user_id)]
    
    def _get_book_id(self, book_idx: int) -> int:
        """Convert matrix index to book ID."""
        return self.id_mappings['idx_to_book'][str(book_idx)]
    
    def get_similar_users(self, user_id: int, n: int = 5) -> List[Tuple[int, float]]:
        """Get N users most similar to the given user."""
        user_idx = self._get_user_idx(user_id)
        user_similarities = self.user_similarity[user_idx]
        
        # Get top N similar users (excluding the user themselves)
        similar_user_idxs = np.argsort(user_similarities)[::-1][1:n+1]
        similar_users = [
            (int(self.id_mappings['idx_to_user'][str(idx)]), 
             user_similarities[idx])
            for idx in similar_user_idxs
        ]
        
        return similar_users
    
    def get_user_recommendations(
        self, 
        user_id: int, 
        top_n: int = 5,
        exclude_listened: bool = True
    ) -> List[Dict]:
        """
        Get top N book recommendations for a user.
        
        Args:
            user_id: ID of the user
            top_n: Number of recommendations to return
            exclude_listened: Whether to exclude books the user has already listened to
            
        Returns:
            List of dictionaries containing recommended book information
        """
        user_idx = self._get_user_idx(user_id)
        
        # Get user's interaction vector and similar users
        user_vector = self.user_item_matrix[user_idx].toarray().flatten()
        similar_users = self.get_similar_users(user_id, n=10)
        
        # Get weighted sum of similar users' interactions
        weighted_sum = np.zeros_like(user_vector)
        weight_sum = np.zeros_like(user_vector)
        
        for similar_user_id, similarity in similar_users:
            similar_user_idx = self._get_user_idx(similar_user_id)
            similar_user_vector = self.user_item_matrix[similar_user_idx].toarray().flatten()
            
            weighted_sum += similarity * similar_user_vector
            weight_sum += np.abs(similarity)
        
        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1
        predicted_ratings = weighted_sum / weight_sum
        
        # If excluding listened books, set their ratings to -1
        if exclude_listened:
            listened_mask = user_vector > 0
            predicted_ratings[listened_mask] = -1
        
        # Get top N book indices
        top_book_idxs = np.argsort(predicted_ratings)[::-1][:top_n]
        
        # Convert to book information
        recommendations = []
        for idx in top_book_idxs:
            book_id = self._get_book_id(idx)
            book_info = self.audiobooks[self.audiobooks['book_id'] == book_id].iloc[0]
            
            recommendations.append({
                'book_id': int(book_id),
                'title': book_info['title'],
                'author': book_info['author'],
                'genre': book_info['genre'],
                'predicted_rating': float(predicted_ratings[idx]),
                'similar_users': [uid for uid, _ in similar_users[:3]]  # Top 3 similar users
            })
        
        return recommendations

def main():
    """Demo the recommender system."""
    # Initialize recommender
    recommender = UserUserCollaborativeFiltering()
    
    # Get recommendations for first user
    user_id = 1
    recommendations = recommender.get_user_recommendations(user_id, top_n=5)
    
    print(f"\nTop 5 recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']} by {rec['author']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Predicted Rating: {rec['predicted_rating']:.2f}")
        print(f"   Based on similar users: {rec['similar_users']}")

if __name__ == "__main__":
    main() 