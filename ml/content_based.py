import numpy as np
import scipy.sparse as sparse
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import List, Dict, Tuple
import pandas as pd

class ContentBasedRecommender:
    def __init__(self, data_dir: str = 'data/processed'):
        """Initialize the content-based recommender system."""
        # Load the content features matrix
        self.content_features = sparse.load_npz(
            os.path.join(data_dir, 'content_features.npz')
        )
        
        # Load feature names
        with open(os.path.join(data_dir, 'feature_names.json'), 'r') as f:
            self.feature_names = json.load(f)
            
        # Load ID mappings
        with open(os.path.join(data_dir, 'id_mappings.json'), 'r') as f:
            self.id_mappings = json.load(f)
            
        # Load audiobooks data
        self.audiobooks = pd.read_csv('data/raw/audiobooks.csv')
        
        # Compute similarity matrix
        self.similarity_matrix = self._compute_similarity_matrix()
        
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute item-item similarity matrix using cosine similarity."""
        return cosine_similarity(self.content_features)
    
    def _get_book_idx(self, book_id: int) -> int:
        """Convert book ID to matrix index."""
        return self.id_mappings['book_to_idx'][str(book_id)]
    
    def _get_book_id(self, book_idx: int) -> int:
        """Convert matrix index to book ID."""
        return self.id_mappings['idx_to_book'][str(book_idx)]
    
    def get_similar_books(
        self, 
        book_id: int, 
        top_n: int = 5,
        include_scores: bool = False
    ) -> List[Dict]:
        """
        Get N most similar books to the given book.
        
        Args:
            book_id: ID of the book to find similar books for
            top_n: Number of similar books to return
            include_scores: Whether to include similarity scores in results
            
        Returns:
            List of dictionaries containing similar book information
        """
        book_idx = self._get_book_idx(book_id)
        similarities = self.similarity_matrix[book_idx]
        
        # Get top N similar books (excluding the book itself)
        similar_idxs = np.argsort(similarities)[::-1][1:top_n+1]
        
        # Get book information
        similar_books = []
        source_book = self.audiobooks[self.audiobooks['book_id'] == book_id].iloc[0]
        
        for idx in similar_idxs:
            book_id = self._get_book_id(idx)
            book_info = self.audiobooks[self.audiobooks['book_id'] == book_id].iloc[0]
            
            recommendation = {
                'book_id': int(book_id),
                'title': book_info['title'],
                'author': book_info['author'],
                'genre': book_info['genre'],
                'similarity_reasons': self._get_similarity_reasons(
                    source_book, book_info
                )
            }
            
            if include_scores:
                recommendation['similarity_score'] = float(similarities[idx])
            
            similar_books.append(recommendation)
        
        return similar_books
    
    def _get_similarity_reasons(
        self, 
        source_book: pd.Series, 
        similar_book: pd.Series,
        max_reasons: int = 3
    ) -> List[str]:
        """Generate human-readable reasons for book similarity."""
        reasons = []
        
        # Check genre similarity
        if source_book['genre'] == similar_book['genre']:
            reasons.append(f"Same genre: {source_book['genre']}")
        
        # Compare tags
        source_tags = set(source_book['tags'].split(', '))
        similar_tags = set(similar_book['tags'].split(', '))
        common_tags = source_tags & similar_tags
        
        if common_tags:
            tags_str = ', '.join(list(common_tags)[:2])
            reasons.append(f"Similar themes: {tags_str}")
            
        # Compare duration
        duration_diff = abs(source_book['duration'] - similar_book['duration'])
        if duration_diff <= 60:  # Within 1 hour
            reasons.append("Similar length")
            
        return reasons[:max_reasons]
    
    def get_recommendations_by_genre(
        self, 
        genre: str, 
        top_n: int = 5
    ) -> List[Dict]:
        """Get top N recommended books in a specific genre."""
        # Get all books in the genre
        genre_books = self.audiobooks[self.audiobooks['genre'] == genre]
        
        if genre_books.empty:
            return []
        
        # Sort by rating and return top N
        top_books = genre_books.nlargest(top_n, 'rating')
        
        recommendations = []
        for _, book in top_books.iterrows():
            recommendations.append({
                'book_id': int(book['book_id']),
                'title': book['title'],
                'author': book['author'],
                'rating': float(book['rating']),
                'tags': book['tags'].split(', ')
            })
            
        return recommendations

def main():
    """Demo the content-based recommender system."""
    recommender = ContentBasedRecommender()
    
    # Get a random book ID to demo
    sample_book_id = recommender.audiobooks['book_id'].iloc[0]
    book_info = recommender.audiobooks[recommender.audiobooks['book_id'] == sample_book_id].iloc[0]
    
    print(f"\nFinding similar books to: {book_info['title']} by {book_info['author']}")
    print(f"Genre: {book_info['genre']}")
    print(f"Tags: {book_info['tags']}")
    
    similar_books = recommender.get_similar_books(
        sample_book_id, 
        top_n=5, 
        include_scores=True
    )
    
    print("\nTop 5 similar books:")
    for i, book in enumerate(similar_books, 1):
        print(f"\n{i}. {book['title']} by {book['author']}")
        print(f"   Genre: {book['genre']}")
        print(f"   Similarity: {book['similarity_score']:.2f}")
        print(f"   Why similar: {', '.join(book['similarity_reasons'])}")
    
    # Demo genre-based recommendations
    genre = book_info['genre']
    print(f"\nTop 5 {genre} books:")
    genre_recommendations = recommender.get_recommendations_by_genre(genre, top_n=5)
    
    for i, book in enumerate(genre_recommendations, 1):
        print(f"\n{i}. {book['title']} by {book['author']}")
        print(f"   Rating: {book['rating']}")
        print(f"   Tags: {', '.join(book['tags'][:3])}")

if __name__ == "__main__":
    main() 