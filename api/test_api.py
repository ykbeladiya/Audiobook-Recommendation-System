import requests
import json
from typing import Dict, List
import pandas as pd

def print_recommendation(rec: Dict) -> None:
    """Pretty print a single recommendation"""
    print(f"\n📚 {rec['title']} by {rec['author']}")
    print(f"   Genre: {rec['genre']}")
    print(f"   Tags: {', '.join(rec['tags'][:3])}")
    print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
    print(f"   - Collaborative Score: {rec['collaborative_score']:.3f}")
    print(f"   - Content Score: {rec['content_score']:.3f}")

def test_recommendations():
    """Test the recommendations endpoint with a real user"""
    # Get a real user_id from the ratings data
    ratings_df = pd.read_csv('data/processed/ratings.csv')
    test_user_id = ratings_df['user_id'].iloc[0]
    
    # API endpoint
    url = f"http://localhost:8000/recommend/{test_user_id}"
    
    try:
        # Make the request
        print(f"\nFetching recommendations for user {test_user_id}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse and display recommendations
        recommendations: List[Dict] = response.json()
        print(f"\nTop {len(recommendations)} Recommendations:")
        
        for rec in recommendations:
            print_recommendation(rec)
            
    except requests.exceptions.ConnectionError:
        print("\n Error: Could not connect to the API server.")
        print("   Make sure the server is running with: uvicorn api.main:app --reload")
    except Exception as e:
        print(f"\n Error: {str(e)}")

if __name__ == "__main__":
    test_recommendations() 
