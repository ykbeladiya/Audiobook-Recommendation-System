"""
Collaborative filtering recommendation model.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilter:
    def __init__(self):
        """Initialize collaborative filtering model."""
        self.user_ratings_matrix = None
        self.user_similarity = None
        self.users = None
        self.books = None
        
    def fit(self, user_ratings: pd.DataFrame):
        """
        Train the collaborative filtering model.
        
        Args:
            user_ratings: DataFrame with columns [user_id, book_id, rating]
        """
        # Create user-item matrix
        self.user_ratings_matrix = user_ratings.pivot(
            index='user_id',
            columns='book_id',
            values='rating'
        ).fillna(0)
        
        # Calculate user similarity matrix
        self.user_similarity = cosine_similarity(self.user_ratings_matrix)
        
        # Store users and books for later use
        self.users = self.user_ratings_matrix.index
        self.books = self.user_ratings_matrix.columns
        
    def predict(self, user_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Predict ratings for unrated items.
        
        Args:
            user_id: ID of the user to get predictions for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (book_id, predicted_rating)
        """
        if self.user_ratings_matrix is None:
            return []
            
        try:
            # Get user index
            user_idx = self.users.get_loc(user_id)
            
            # Get user's ratings and similar users
            user_ratings = self.user_ratings_matrix.iloc[user_idx]
            similar_users = self.user_similarity[user_idx]
            
            # Get indices of books user hasn't rated
            unrated_books = user_ratings[user_ratings == 0].index
            
            if len(unrated_books) == 0:
                return []
                
            # Calculate predicted ratings
            predictions = []
            for book_id in unrated_books:
                book_idx = self.books.get_loc(book_id)
                # Get ratings for this book from other users
                other_ratings = self.user_ratings_matrix.iloc[:, book_idx]
                # Calculate weighted average rating
                weighted_sum = np.sum(other_ratings * similar_users)
                similarity_sum = np.sum(similar_users)
                predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 0
                predictions.append((book_id, predicted_rating))
            
            # Sort by predicted rating and return top N
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:n_recommendations]
            
        except (KeyError, ValueError) as e:
            print(f"Error predicting for user {user_id}: {str(e)}")
            return [] 