import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

class HybridRecommender:
    def __init__(self, collaborative_weight=0.6, content_weight=0.4):
        """
        Initialize the hybrid recommender system.
        
        Args:
            collaborative_weight (float): Weight for collaborative filtering (default: 0.6)
            content_weight (float): Weight for content-based filtering (default: 0.4)
        """
        self.collaborative_weight = collaborative_weight
        self.content_weight = content_weight
        self.books_df = None
        self.ratings_df = None
        self.user_item_matrix = None
        self.content_similarity_matrix = None
        
    def load_data(self, books_path, ratings_path):
        """
        Load books and ratings data from CSV files.
        
        Args:
            books_path (str): Path to the books CSV file
            ratings_path (str): Path to the ratings CSV file
        """
        self.books_df = pd.read_csv(books_path)
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Create user-item matrix for collaborative filtering
        self.user_item_matrix = self.ratings_df.pivot(
            index='user_id', 
            columns='book_id', 
            values='rating'
        ).fillna(0)
        
        # Create content-based similarity matrix
        self._create_content_similarity_matrix()
        
    def _create_content_similarity_matrix(self):
        """Create content-based similarity matrix using TF-IDF on book metadata."""
        # Combine relevant features for content-based filtering
        self.books_df['content_features'] = self.books_df.apply(
            lambda x: f"{x['title']} {x['author']} {x['genre']} {x['description']} {x['tags']}",
            axis=1
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.books_df['content_features'])
        
        # Calculate cosine similarity
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
    def _get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """
        Get collaborative filtering recommendations for a user using item-item similarity.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of book_ids and their predicted ratings
        """
        if user_id not in self.user_item_matrix.index:
            return {}
            
        # Calculate item-item similarity
        item_similarity = cosine_similarity(self.user_item_matrix.T)
        
        # Get user's ratings
        user_ratings = self.user_item_matrix.loc[user_id]
        
        # Calculate predicted ratings
        predictions = {}
        for book_id in self.books_df['book_id']:
            if book_id not in self.user_item_matrix.columns:
                continue
                
            book_idx = self.user_item_matrix.columns.get_loc(book_id)
            similar_items = item_similarity[book_idx]
            
            # Calculate predicted rating
            weighted_sum = np.sum(similar_items * user_ratings)
            similarity_sum = np.sum(np.abs(similar_items))
            
            if similarity_sum != 0:
                predictions[book_id] = weighted_sum / similarity_sum
            
        return dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
    
    def _get_content_recommendations(self, user_id, n_recommendations=10):
        """
        Get content-based recommendations based on user's highly rated books.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            dict: Dictionary of book_ids and their similarity scores
        """
        # Get user's top rated books
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        top_rated = user_ratings.nlargest(5, 'rating')
        
        similar_books = {}
        for _, row in top_rated.iterrows():
            book_idx = self.books_df[self.books_df['book_id'] == row['book_id']].index[0]
            similarities = self.content_similarity_matrix[book_idx]
            
            for idx, score in enumerate(similarities):
                book_id = self.books_df.iloc[idx]['book_id']
                if book_id not in similar_books:
                    similar_books[book_id] = score * row['rating']
                else:
                    similar_books[book_id] = max(similar_books[book_id], score * row['rating'])
        
        return dict(sorted(similar_books.items(), key=lambda x: x[1], reverse=True)[:n_recommendations])
    
    def _get_user_listened_books(self, user_id):
        """Get the set of books already listened to by the user."""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        return set(user_ratings['book_id'])
    
    def generate_recommendations(self, user_id, n_recommendations=10):
        """
        Generate hybrid recommendations for a user.
        
        Args:
            user_id (int): User ID
            n_recommendations (int): Number of recommendations to return
            
        Returns:
            list: List of dictionaries containing recommended books and their scores
        """
        # Get recommendations from both systems
        collab_recs = self._get_collaborative_recommendations(user_id, n_recommendations * 2)
        content_recs = self._get_content_recommendations(user_id, n_recommendations * 2)
        
        # Normalize scores
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {}
            values = list(scores_dict.values())
            min_val, max_val = min(values), max(values)
            return {k: (v - min_val) / (max_val - min_val) if max_val > min_val else v 
                   for k, v in scores_dict.items()}
        
        collab_recs = normalize_scores(collab_recs)
        content_recs = normalize_scores(content_recs)
        
        # Combine recommendations with weights
        hybrid_scores = {}
        all_book_ids = set(collab_recs.keys()) | set(content_recs.keys())
        
        for book_id in all_book_ids:
            collab_score = collab_recs.get(book_id, 0)
            content_score = content_recs.get(book_id, 0)
            hybrid_scores[book_id] = (collab_score * self.collaborative_weight + 
                                    content_score * self.content_weight)
        
        # Remove already listened books
        listened_books = self._get_user_listened_books(user_id)
        hybrid_scores = {k: v for k, v in hybrid_scores.items() if k not in listened_books}
        
        # Sort and get top recommendations
        top_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        
        # Format recommendations with book details
        recommendations = []
        for book_id, score in top_recommendations:
            book_info = self.books_df[self.books_df['book_id'] == book_id].iloc[0]
            recommendations.append({
                'book_id': book_id,
                'title': book_info['title'],
                'author': book_info['author'],
                'genre': book_info['genre'],
                'score': score,
                'publication_year': book_info['publication_year']
            })
        
        return recommendations 