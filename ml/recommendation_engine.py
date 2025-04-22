from typing import List, Dict, Set
import numpy as np
import pandas as pd
from ml.collaborative_filtering import UserUserCollaborativeFiltering
from ml.content_based import ContentBasedRecommender

class HybridRecommender:
    def __init__(
        self,
        collaborative_weight: float = 0.6,
        content_weight: float = 0.4,
        data_dir: str = 'data/processed'
    ):
        """
        Initialize hybrid recommender system.
        
        Args:
            collaborative_weight: Weight for collaborative filtering scores (0-1)
            content_weight: Weight for content-based scores (0-1)
            data_dir: Directory containing processed data files
        """
        if not np.isclose(collaborative_weight + content_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
            
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        
        # Initialize both recommenders
        self.collaborative = UserUserCollaborativeFiltering(data_dir)
        self.content_based = ContentBasedRecommender(data_dir)
        
        # Load audiobooks data
        self.audiobooks = pd.read_csv('data/raw/audiobooks.csv')
    
    def _get_user_listened_books(self, user_id: int) -> Set[int]:
        """Get set of books the user has already listened to."""
        user_idx = self.collaborative._get_user_idx(user_id)
        user_vector = self.collaborative.user_item_matrix[user_idx].toarray().flatten()
        listened_idxs = np.where(user_vector > 0)[0]
        
        return {
            self.collaborative._get_book_id(idx)
            for idx in listened_idxs
        }
    
    def _get_collaborative_recommendations(
        self,
        user_id: int,
        n: int = 10
    ) -> Dict[int, float]:
        """Get collaborative filtering recommendations with scores."""
        recommendations = self.collaborative.get_user_recommendations(
            user_id,
            top_n=n,
            exclude_listened=True
        )
        
        return {
            rec['book_id']: rec['predicted_rating']
            for rec in recommendations
        }
    
    def _get_content_recommendations(
        self,
        user_id: int,
        n: int = 10
    ) -> Dict[int, float]:
        """
        Get content-based recommendations based on user's listening history.
        Uses average similarity to user's top-rated books.
        """
        # Get user's listened books and their ratings
        user_idx = self.collaborative._get_user_idx(user_id)
        user_vector = self.collaborative.user_item_matrix[user_idx].toarray().flatten()
        
        # Get user's top 3 highest-rated books
        top_book_idxs = np.argsort(user_vector)[::-1][:3]
        top_book_ids = [
            self.collaborative._get_book_id(idx)
            for idx in top_book_idxs
        ]
        
        # Get similar books for each top book
        all_similar_books = {}
        for book_id in top_book_ids:
            similar_books = self.content_based.get_similar_books(
                book_id,
                top_n=n,
                include_scores=True
            )
            
            for book in similar_books:
                book_id = book['book_id']
                if book_id not in all_similar_books:
                    all_similar_books[book_id] = []
                all_similar_books[book_id].append(book['similarity_score'])
        
        # Average similarity scores for books recommended multiple times
        return {
            book_id: np.mean(scores)
            for book_id, scores in all_similar_books.items()
        }
    
    def generate_recommendations(
        self,
        user_id: int,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            top_n: Number of recommendations to return
            
        Returns:
            List of dictionaries containing recommended book information
        """
        # Get recommendations from both approaches
        collaborative_recs = self._get_collaborative_recommendations(user_id, n=top_n*2)
        content_recs = self._get_content_recommendations(user_id, n=top_n*2)
        
        # Normalize scores to 0-1 range for each method
        def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
            if not scores:
                return {}
            min_score = min(scores.values())
            max_score = max(scores.values())
            score_range = max_score - min_score
            if score_range == 0:
                return {k: 1.0 for k in scores}
            return {
                k: (v - min_score) / score_range
                for k, v in scores.items()
            }
        
        collab_scores = normalize_scores(collaborative_recs)
        content_scores = normalize_scores(content_recs)
        
        # Combine scores with weights
        all_book_ids = set(collab_scores.keys()) | set(content_scores.keys())
        hybrid_scores = {}
        
        for book_id in all_book_ids:
            collab_score = collab_scores.get(book_id, 0.0)
            content_score = content_scores.get(book_id, 0.0)
            
            hybrid_scores[book_id] = (
                self.collaborative_weight * collab_score +
                self.content_weight * content_score
            )
        
        # Get top N recommendations
        listened_books = self._get_user_listened_books(user_id)
        top_books = sorted(
            [
                (book_id, score)
                for book_id, score in hybrid_scores.items()
                if book_id not in listened_books
            ],
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Get recommendation details
        recommendations = []
        for book_id, hybrid_score in top_books:
            book_info = self.audiobooks[self.audiobooks['book_id'] == book_id].iloc[0]
            
            recommendations.append({
                'book_id': int(book_id),
                'title': book_info['title'],
                'author': book_info['author'],
                'genre': book_info['genre'],
                'hybrid_score': float(hybrid_score),
                'collaborative_score': float(collab_scores.get(book_id, 0.0)),
                'content_score': float(content_scores.get(book_id, 0.0)),
                'tags': book_info['tags'].split(', ')
            })
        
        return recommendations

def main():
    """Demo the hybrid recommender system."""
    recommender = HybridRecommender()
    
    # Demo recommendations for first user
    user_id = 1
    recommendations = recommender.generate_recommendations(user_id, top_n=5)
    
    print(f"\nTop 5 hybrid recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['title']} by {rec['author']}")
        print(f"   Genre: {rec['genre']}")
        print(f"   Tags: {', '.join(rec['tags'][:3])}")
        print(f"   Hybrid Score: {rec['hybrid_score']:.2f}")
        print(f"   - Collaborative Score: {rec['collaborative_score']:.2f}")
        print(f"   - Content Score: {rec['content_score']:.2f}")

if __name__ == "__main__":
    main() 