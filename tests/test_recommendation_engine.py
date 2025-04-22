import pytest
import numpy as np
import pandas as pd
from decimal import Decimal
from unittest.mock import Mock, patch
from src.recommendation_engine import HybridRecommender
from src.models.collaborative_filtering import CollaborativeFilter
from src.models.content_based import ContentBasedRecommender

@pytest.fixture
def sample_user_ratings():
    """Fixture for sample user ratings data."""
    return pd.DataFrame({
        'user_id': ['1', '1', '2', '2', '3'],
        'book_id': ['101', '102', '101', '103', '102'],
        'rating': [4.5, 3.0, 5.0, 4.0, 3.5],
        'timestamp': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03', '2024-01-02']
    })

@pytest.fixture
def sample_books():
    """Fixture for sample books data."""
    return pd.DataFrame({
        'book_id': ['101', '102', '103', '104', '105'],
        'title': ['Book 1', 'Book 2', 'Book 3', 'Book 4', 'Book 5'],
        'author': ['Author 1', 'Author 2', 'Author 1', 'Author 3', 'Author 2'],
        'genre': ['Fiction', 'Mystery', 'Fiction', 'Romance', 'Mystery']
    })

@pytest.fixture
def mock_collaborative_filter():
    """Fixture for mocked collaborative filter."""
    mock = Mock(spec=CollaborativeFilter)
    mock.predict.return_value = [
        ('104', 4.8),
        ('105', 4.2),
        ('103', 3.9)
    ]
    return mock

@pytest.fixture
def mock_content_based():
    """Fixture for mocked content-based recommender."""
    mock = Mock(spec=ContentBasedRecommender)
    mock.get_similar_books.return_value = [
        ('105', 0.95),
        ('103', 0.85),
        ('104', 0.75)
    ]
    return mock

@pytest.fixture
def recommender(mock_collaborative_filter, mock_content_based, sample_user_ratings, sample_books):
    """Fixture for hybrid recommender with mocked components."""
    recommender = HybridRecommender(
        collaborative_weight=0.6,
        content_weight=0.4
    )
    recommender.collaborative_filter = mock_collaborative_filter
    recommender.content_based = mock_content_based
    recommender.user_ratings = sample_user_ratings
    recommender.books = sample_books
    return recommender

def test_get_user_recommendations(recommender):
    """Test getting recommendations for a specific user."""
    user_id = '1'
    n_recommendations = 3
    
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == n_recommendations
    assert all(isinstance(rec['book_id'], str) for rec in recommendations)
    assert all(isinstance(rec['score'], float) for rec in recommendations)
    assert all(rec['score'] >= 0 and rec['score'] <= 5 for rec in recommendations)
    
    # Check that recommended books weren't already rated by the user
    user_rated_books = set(recommender.user_ratings[
        recommender.user_ratings['user_id'] == user_id
    ]['book_id'])
    recommended_books = set(rec['book_id'] for rec in recommendations)
    assert not (user_rated_books & recommended_books)

def test_get_user_recommendations_no_ratings(recommender):
    """Test getting recommendations for a user with no ratings."""
    user_id = '999'  # Non-existent user
    n_recommendations = 3
    
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == n_recommendations
    recommender.collaborative_filter.predict.assert_called_once_with(user_id, n_recommendations)

def test_get_user_recommendations_fewer_available(recommender):
    """Test when fewer recommendations are available than requested."""
    user_id = '1'
    n_recommendations = 10  # More than available
    
    # Mock fewer recommendations
    recommender.collaborative_filter.predict.return_value = [('104', 4.8), ('105', 4.2)]
    
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == 2  # Should return all available recommendations
    assert all(isinstance(rec['book_id'], str) for rec in recommendations)

def test_get_similar_books(recommender):
    """Test getting similar books based on content."""
    book_id = '101'
    n_recommendations = 3
    
    similar_books = recommender.get_similar_books(book_id, n_recommendations)
    
    assert len(similar_books) == n_recommendations
    assert all(isinstance(book['book_id'], str) for book in similar_books)
    assert all(isinstance(book['similarity'], float) for book in similar_books)
    assert all(book['similarity'] >= 0 and book['similarity'] <= 1 
              for book in similar_books)
    assert book_id not in [book['book_id'] for book in similar_books]

def test_get_similar_books_invalid_book(recommender):
    """Test getting similar books for an invalid book ID."""
    book_id = 'invalid_id'
    n_recommendations = 3
    
    # Mock empty response for invalid book
    recommender.content_based.get_similar_books.return_value = []
    
    similar_books = recommender.get_similar_books(book_id, n_recommendations)
    
    assert len(similar_books) == 0
    recommender.content_based.get_similar_books.assert_called_once_with(book_id, n_recommendations)

def test_get_similar_books_exact_matches(recommender):
    """Test getting similar books with exact similarity scores."""
    book_id = '101'
    n_recommendations = 3
    
    # Mock exact similarity scores
    recommender.content_based.get_similar_books.return_value = [
        ('102', 1.0),
        ('103', 1.0),
        ('104', 1.0)
    ]
    
    similar_books = recommender.get_similar_books(book_id, n_recommendations)
    
    assert len(similar_books) == n_recommendations
    assert all(book['similarity'] == 1.0 for book in similar_books)

def test_generate_recommendations(recommender):
    """Test generating hybrid recommendations."""
    user_id = '1'
    n_recommendations = 3
    
    recommendations = recommender.generate_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == n_recommendations
    assert all(isinstance(rec['book_id'], str) for rec in recommendations)
    assert all(isinstance(rec['score'], float) for rec in recommendations)
    assert all(rec['score'] >= 0 and rec['score'] <= 5 for rec in recommendations)
    
    # Verify recommendations are sorted by score in descending order
    scores = [rec['score'] for rec in recommendations]
    assert scores == sorted(scores, reverse=True)

def test_generate_recommendations_weight_balance(recommender):
    """Test that recommendations properly balance collaborative and content weights."""
    user_id = '1'
    n_recommendations = 1
    
    # Mock same book_id with different scores
    recommender.collaborative_filter.predict.return_value = [('104', 5.0)]
    recommender.content_based.get_similar_books.return_value = [('104', 0.0)]
    
    recommendations = recommender.generate_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == 1
    # Expected score: 0.6 * 5.0 + 0.4 * 0.0 = 3.0
    assert abs(recommendations[0]['score'] - 3.0) < 0.01

def test_generate_recommendations_unique_results(recommender):
    """Test that recommendations don't contain duplicates."""
    user_id = '1'
    n_recommendations = 5
    
    # Mock overlapping recommendations
    recommender.collaborative_filter.predict.return_value = [
        ('104', 4.8),
        ('105', 4.2),
        ('103', 3.9)
    ]
    recommender.content_based.get_similar_books.return_value = [
        ('104', 0.95),  # Duplicate
        ('106', 0.85),
        ('107', 0.75)
    ]
    
    recommendations = recommender.generate_recommendations(user_id, n_recommendations)
    
    # Check for unique book_ids
    book_ids = [rec['book_id'] for rec in recommendations]
    assert len(book_ids) == len(set(book_ids))

def test_recommendations_with_no_history(recommender):
    """Test recommendations for a new user with no rating history."""
    user_id = '999'  # Non-existent user
    n_recommendations = 3
    
    recommendations = recommender.generate_recommendations(user_id, n_recommendations)
    
    assert len(recommendations) == n_recommendations
    # Should fall back to content-based recommendations
    recommender.content_based.get_similar_books.assert_called_once()

def test_recommendations_with_invalid_weights():
    """Test that invalid weights raise ValueError."""
    with pytest.raises(ValueError):
        HybridRecommender(collaborative_weight=0.7, content_weight=0.5)
    
    with pytest.raises(ValueError):
        HybridRecommender(collaborative_weight=-0.1, content_weight=1.1)

def test_empty_recommendations(recommender):
    """Test handling of empty recommendation lists."""
    recommender.collaborative_filter.predict.return_value = []
    recommender.content_based.get_similar_books.return_value = []
    
    recommendations = recommender.generate_recommendations('1', 3)
    
    assert len(recommendations) == 0

def test_recommendations_normalization(recommender):
    """Test that recommendation scores are properly normalized."""
    user_id = '1'
    n_recommendations = 3
    
    recommendations = recommender.generate_recommendations(user_id, n_recommendations)
    
    # Check if scores are properly weighted and normalized
    assert all(0 <= rec['score'] <= 5 for rec in recommendations)
    
    # Verify collaborative and content weights are properly applied
    collab_weight = recommender.collaborative_weight
    content_weight = recommender.content_weight
    
    # Get raw scores
    collab_scores = dict(recommender.collaborative_filter.predict())
    content_scores = dict(recommender.content_based.get_similar_books())
    
    # Check first recommendation's score
    first_rec = recommendations[0]
    expected_score = (
        collab_weight * collab_scores.get(first_rec['book_id'], 0) +
        content_weight * content_scores.get(first_rec['book_id'], 0)
    )
    assert abs(first_rec['score'] - expected_score) < 0.01

def test_get_user_listened_books(recommender):
    """Test getting the list of books a user has already listened to."""
    user_id = '1'
    
    listened_books = recommender._get_user_listened_books(user_id)
    
    assert isinstance(listened_books, set)
    assert len(listened_books) > 0
    assert all(isinstance(book_id, str) for book_id in listened_books)
    assert listened_books == set(recommender.user_ratings[
        recommender.user_ratings['user_id'] == user_id
    ]['book_id']) 