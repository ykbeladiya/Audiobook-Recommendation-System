"""
Streamlit app for Audiobook Recommendations.
"""
import streamlit as st
import requests
import pandas as pd
from typing import List, Dict, Optional

# Constants
API_BASE_URL = "http://localhost:8000"  # Update this with your FastAPI endpoint
DEFAULT_LIMIT = 5

# Load book metadata
try:
    books_df = pd.read_csv("data/raw/books.csv")
except Exception as e:
    st.error(f"Error loading book metadata: {str(e)}")
    books_df = pd.DataFrame()

def get_book_metadata(book_id: str) -> Dict:
    """Get book metadata from the DataFrame."""
    try:
        book = books_df[books_df['book_id'] == book_id].iloc[0]
        return {
            'title': book['title'],
            'author': book['author'],
            'genre': book['genre'],
            'description': book.get('description', 'No description available')
        }
    except (IndexError, KeyError):
        return {
            'title': 'Unknown Book',
            'author': 'Unknown Author',
            'genre': 'Unknown Genre',
            'description': 'Metadata not available'
        }

def fetch_recommendations(endpoint: str, params: Dict) -> List[Dict]:
    """Fetch recommendations from the API endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching recommendations: {str(e)}")
        return []

def display_recommendations(recommendations: List[Dict]):
    """Display recommendations in a nice format."""
    if not recommendations:
        st.warning("No recommendations found.")
        return

    for rec in recommendations:
        book_id = rec.get('book_id')
        score = rec.get('score', 0)
        metadata = get_book_metadata(book_id)
        
        with st.expander(f"{metadata['title']} (Score: {score:.2f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Author:**", metadata['author'])
                st.write("**Book ID:**", book_id)
            with col2:
                st.write("**Genre:**", metadata['genre'])
            st.write("**Description:**", metadata['description'])

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Audiobook Recommendations",
        page_icon="🎧",
        layout="wide"
    )
    
    st.title("📚 Audiobook Recommendation System")
    
    # Sidebar
    st.sidebar.title("Navigation")

    # Recommendation type selector
    rec_type = st.sidebar.selectbox(
        "Select Recommendation Type",
        ["User-Based", "Similar Books", "Hybrid"]
    )

    if rec_type == "User-Based":
        st.header("User-Based Recommendations")
        user_id = st.text_input("Enter User ID")
        
        if user_id:
            with st.spinner("Fetching recommendations..."):
                recommendations = fetch_recommendations(
                    f"recommendations/user/{user_id}",
                    {"limit": DEFAULT_LIMIT}
                )
                display_recommendations(recommendations)

    elif rec_type == "Similar Books":
        st.header("Similar Books Recommendations")
        if not books_df.empty:
            book_options = books_df[['book_id', 'title']].apply(
                lambda x: f"{x['title']} (ID: {x['book_id']})",
                axis=1
            ).tolist()
            
            selected_book = st.selectbox("Select a Book", book_options)
            if selected_book:
                book_id = selected_book.split("(ID: ")[-1].rstrip(")")
                with st.spinner("Fetching similar books..."):
                    recommendations = fetch_recommendations(
                        f"recommendations/similar/{book_id}",
                        {"limit": DEFAULT_LIMIT}
                    )
                    display_recommendations(recommendations)
        else:
            st.error("Book metadata not available. Cannot show book selection.")

    else:  # Hybrid
        st.header("Hybrid Recommendations")
        user_id = st.text_input("Enter User ID")
        
        if user_id:
            with st.spinner("Fetching hybrid recommendations..."):
                recommendations = fetch_recommendations(
                    f"recommendations/hybrid/{user_id}",
                    {"limit": DEFAULT_LIMIT}
                )
                display_recommendations(recommendations)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Made with ❤️ by Your Team"
    )

if __name__ == "__main__":
    main() 