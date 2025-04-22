from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import pandas as pd
from ml.recommendation_engine import HybridRecommender
from pydantic import BaseModel
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Initialize FastAPI app
app = FastAPI(
    title="Audiobook Recommendation API",
    description="API for generating audiobook recommendations using hybrid filtering",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize recommender system and load book data
try:
    recommender = HybridRecommender()
    books_df = pd.read_csv('data/raw/audiobooks.csv')
except Exception as e:
    print(f"Error initializing the system: {str(e)}")
    raise

class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: Optional[int] = 10
    collaborative_weight: Optional[float] = 0.6
    content_weight: Optional[float] = 0.4

class RecommendationResponse(BaseModel):
    book_id: int
    title: str
    author: str
    genre: str
    score: float
    publication_year: int

@app.on_event("startup")
async def startup_event():
    """Initialize the recommender system on startup"""
    books_path = os.path.join(project_root, "data", "raw", "books.csv")
    ratings_path = os.path.join(project_root, "data", "raw", "user_ratings.csv")
    
    try:
        recommender.load_data(books_path, ratings_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise RuntimeError("Failed to initialize recommender system")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Audiobook Recommendation API"}

@app.post("/recommendations/", response_model=List[RecommendationResponse])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized audiobook recommendations for a user
    
    - **user_id**: The ID of the user to get recommendations for
    - **num_recommendations**: Number of recommendations to return (default: 10)
    - **collaborative_weight**: Weight for collaborative filtering (default: 0.6)
    - **content_weight**: Weight for content-based filtering (default: 0.4)
    """
    try:
        # Update weights if provided
        if request.collaborative_weight != recommender.collaborative_weight or \
           request.content_weight != recommender.content_weight:
            recommender.collaborative_weight = request.collaborative_weight
            recommender.content_weight = request.content_weight
        
        # Get recommendations
        recommendations = recommender.generate_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user {request.user_id}"
            )
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "recommender_initialized": recommender.books_df is not None and recommender.ratings_df is not None
    }

@app.get("/book/{book_id}")
async def get_book_metadata(book_id: int) -> Dict:
    """
    Get metadata for a specific book
    
    Args:
        book_id: The ID of the book to get metadata for
        
    Returns:
        Dictionary containing book metadata
    """
    try:
        book = books_df[books_df['book_id'] == book_id].iloc[0]
        return {
            'book_id': int(book_id),
            'title': book['title'],
            'author': book['author'],
            'genre': book['genre'],
            'description': book['description'],
            'tags': book['tags'].split(', '),
            'average_rating': float(book['average_rating']),
            'total_ratings': int(book['total_ratings']),
            'duration_hours': float(book['duration_hours']),
            'narrator': book['narrator'],
            'release_date': book['release_date']
        }
    except IndexError:
        raise HTTPException(
            status_code=404,
            detail=f"Book {book_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching book metadata: {str(e)}"
        )

# Exception handler for generic exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 