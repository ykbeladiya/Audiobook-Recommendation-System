"""
Main FastAPI application module.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .utils.logger import setup_logging, logger
from .recommendation_engine import HybridRecommender
from typing import List, Dict, Any
import pandas as pd

app = FastAPI(
    title="Audiobook Recommendation API",
    description="API for getting audiobook recommendations",
    version="1.0.0"
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up logging
setup_logging(app)

# Initialize recommender and load data
recommender = HybridRecommender()
try:
    # Load data
    recommender.user_ratings = pd.read_csv("data/raw/user_ratings.csv")
    recommender.books = pd.read_csv("data/raw/books.csv")
    logger.info("Successfully loaded user ratings and books data")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise RuntimeError(f"Failed to load required data: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.get("/recommendations/user/{user_id}")
async def get_user_recommendations(
    user_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get personalized recommendations for a user."""
    try:
        logger.info(f"Getting recommendations for user {user_id}")
        recommendations = recommender.get_user_recommendations(user_id, limit)
        logger.info(f"Found {len(recommendations)} recommendations for user {user_id}")
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting recommendations: {str(e)}"
        )

@app.get("/recommendations/similar/{book_id}")
async def get_similar_books(
    book_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get similar books recommendations."""
    try:
        logger.info(f"Getting similar books for book {book_id}")
        similar_books = recommender.get_similar_books(book_id, limit)
        logger.info(f"Found {len(similar_books)} similar books for book {book_id}")
        return similar_books
    except Exception as e:
        logger.error(f"Error getting similar books for book {book_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting similar books: {str(e)}"
        )

@app.get("/recommendations/hybrid/{user_id}")
async def get_hybrid_recommendations(
    user_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """Get hybrid recommendations combining collaborative and content-based approaches."""
    try:
        logger.info(f"Getting hybrid recommendations for user {user_id}")
        recommendations = recommender.generate_recommendations(user_id, limit)
        logger.info(f"Found {len(recommendations)} hybrid recommendations for user {user_id}")
        return recommendations
    except Exception as e:
        logger.error(f"Error getting hybrid recommendations for user {user_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting hybrid recommendations: {str(e)}"
        ) 