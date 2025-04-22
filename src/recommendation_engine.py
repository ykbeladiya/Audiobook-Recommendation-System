"""
Hybrid recommendation engine combining collaborative and content-based filtering.
"""
from typing import List, Dict, Any
import pandas as pd
from .models.collaborative_filtering import CollaborativeFilter
from .models.content_based import ContentBasedRecommender

class HybridRecommender:
    def __init__(self, collaborative_weight: float = 0.6, content_weight: float = 0.4):
        """
        Initialize hybrid recommender.
        
        Args:
            collaborative_weight: Weight for collaborative filtering recommendations
            content_weight: Weight for content-based recommendations
        """
        if not (0 <= collaborative_weight <= 1 and 0 <= content_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if abs(collaborative_weight + content_weight - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.collaborative_filter = CollaborativeFilter()
        self.content_based = ContentBasedRecommender()
        self._user_ratings = None
        self._books = None
        
    @property
    def user_ratings(self) -> pd.DataFrame:
        return self._user_ratings
        
    @user_ratings.setter
    def user_ratings(self, df: pd.DataFrame):
        self._user_ratings = df
        if df is not None and self._books is not None:
            self._fit_models()
            
    @property
    def books(self) -> pd.DataFrame:
        return self._books
        
    @books.setter
    def books(self, df: pd.DataFrame):
        self._books = df
        if df is not None and self._user_ratings is not None:
            self._fit_models()
            
    def _fit_models(self):
        """Fit both recommendation models with current data."""
        if self._user_ratings is not None:
            self.collaborative_filter.fit(self._user_ratings)
        if self._books is not None:
            self.content_based.fit(self._books)
        
    def _get_user_listened_books(self, user_id: str) -> set:
        """Get set of books the user has already listened to."""
        if self._user_ratings is None:
            return set()
        return set(self._user_ratings[
            self._user_ratings['user_id'] == user_id
        ]['book_id'])
        
    def get_user_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommendations for a specific user.
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries containing book_id and score
        """
        # Get collaborative filtering recommendations
        collab_recs = self.collaborative_filter.predict(user_id, n_recommendations)
        
        # Convert to list of dictionaries
        recommendations = [
            {'book_id': book_id, 'score': score}
            for book_id, score in collab_recs
        ]
        
        # Filter out books the user has already listened to
        listened_books = self._get_user_listened_books(user_id)
        recommendations = [
            rec for rec in recommendations
            if rec['book_id'] not in listened_books
        ]
        
        return recommendations[:n_recommendations]
        
    def get_similar_books(self, book_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar books based on content.
        
        Args:
            book_id: ID of the book to find similar books for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries containing book_id and similarity score
        """
        similar_books = self.content_based.get_similar_books(book_id, n_recommendations)
        
        return [
            {'book_id': similar_id, 'score': score}
            for similar_id, score in similar_books
            if similar_id != book_id
        ]
        
    def generate_recommendations(self, user_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate hybrid recommendations combining both approaches.
        
        Args:
            user_id: ID of the user to get recommendations for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of dictionaries containing book_id and normalized score
        """
        # Get recommendations from both models
        collab_recs = dict(self.collaborative_filter.predict(user_id, n_recommendations * 2))
        
        # Get user's top-rated books for content-based recommendations
        if self._user_ratings is not None:
            user_books = self._user_ratings[
                self._user_ratings['user_id'] == user_id
            ].sort_values('rating', ascending=False)
            
            if not user_books.empty:
                top_book_id = user_books.iloc[0]['book_id']
                content_recs = dict(self.content_based.get_similar_books(top_book_id, n_recommendations * 2))
            else:
                content_recs = {}
        else:
            content_recs = {}
        
        # Combine all unique book IDs
        all_book_ids = set(collab_recs.keys()) | set(content_recs.keys())
        
        # Filter out books the user has already listened to
        listened_books = self._get_user_listened_books(user_id)
        all_book_ids = all_book_ids - listened_books
        
        if not all_book_ids:
            return []
            
        # Calculate weighted scores
        recommendations = []
        for book_id in all_book_ids:
            collab_score = collab_recs.get(book_id, 0)
            content_score = content_recs.get(book_id, 0)
            
            # Calculate weighted average
            score = (
                self.collaborative_weight * collab_score +
                self.content_weight * content_score
            )
            
            recommendations.append({
                'book_id': book_id,
                'score': score
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations] 