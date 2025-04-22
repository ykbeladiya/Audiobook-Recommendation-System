"""
Content-based recommendation model.
"""
from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        """Initialize content-based recommender."""
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.book_features = None
        self.books_df = None
        
    def fit(self, books_df: pd.DataFrame):
        """
        Train the content-based recommender.
        
        Args:
            books_df: DataFrame with book metadata
        """
        self.books_df = books_df
        
        # Combine relevant features into a single text
        books_df['content'] = books_df.apply(
            lambda x: ' '.join([
                str(x['title']),
                str(x['author']),
                str(x['genre']),
                str(x.get('description', ''))
            ]).lower(),
            axis=1
        )
        
        # Create TF-IDF matrix
        self.book_features = self.tfidf.fit_transform(books_df['content'])
        
    def get_similar_books(self, book_id: str, n_recommendations: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar books based on content.
        
        Args:
            book_id: ID of the book to find similar books for
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of tuples (book_id, similarity_score)
        """
        if self.book_features is None or self.books_df is None:
            return []
            
        try:
            # Get book index
            book_idx = self.books_df[self.books_df['book_id'] == book_id].index[0]
            
            # Calculate similarity scores
            book_vector = self.book_features[book_idx]
            similarity_scores = cosine_similarity(book_vector, self.book_features).flatten()
            
            # Get indices of similar books (excluding the input book)
            similar_indices = np.argsort(similarity_scores)[::-1][1:n_recommendations+1]
            
            # Get book IDs and scores
            recommendations = [
                (
                    self.books_df.iloc[idx]['book_id'],
                    float(similarity_scores[idx])
                )
                for idx in similar_indices
            ]
            
            return recommendations
            
        except (IndexError, KeyError) as e:
            print(f"Error finding similar books for {book_id}: {str(e)}")
            return [] 