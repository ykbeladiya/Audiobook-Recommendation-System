import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict

# Constants for data generation
GENRES = [
    "Mystery", "Science Fiction", "Fantasy", "Romance", "Thriller",
    "Historical Fiction", "Biography", "Self-Help", "Business", "Literary Fiction"
]

TAGS = [
    "bestseller", "award-winning", "page-turner", "thought-provoking",
    "emotional", "complex-plot", "character-driven", "atmospheric",
    "inspiring", "educational", "romantic", "suspenseful", "action-packed",
    "philosophical", "dark", "uplifting", "classic", "contemporary"
]

# Word lists for generating names and titles
ADJECTIVES = [
    "Silent", "Dark", "Bright", "Lost", "Hidden", "Secret", "Ancient", "Modern",
    "Eternal", "Wild", "Gentle", "Fierce", "Quiet", "Loud", "Mysterious", "Sacred",
    "Forgotten", "Endless", "Broken", "Healing", "Rising", "Falling", "Golden"
]

NOUNS = [
    "Shadow", "Light", "Storm", "River", "Mountain", "Forest", "City", "Ocean",
    "Sky", "Star", "Moon", "Sun", "Heart", "Mind", "Soul", "Spirit", "World",
    "Dream", "Memory", "Time", "Path", "Journey", "Story", "Song", "Dance"
]

FIRST_NAMES = [
    "James", "Emma", "William", "Olivia", "Alexander", "Sophia", "Michael",
    "Isabella", "Benjamin", "Ava", "Daniel", "Mia", "Joseph", "Charlotte",
    "David", "Amelia", "John", "Elizabeth", "Samuel", "Sofia"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]

def generate_description(title: str, author: str, genre: str, tags: List[str]) -> str:
    """Generate a realistic book description."""
    templates = [
        f"A captivating {genre.lower()} novel by {author}, '{title}' {random.choice(['explores', 'delves into', 'investigates'])} "
        f"the {random.choice(['fascinating', 'complex', 'intriguing'])} world of {random.choice(['human relationships', 'personal growth', 'adventure'])}.",
        
        f"In this {random.choice(tags)} {genre.lower()} masterpiece, {author} {random.choice(['weaves', 'crafts', 'creates'])} "
        f"a {random.choice(['spellbinding', 'compelling', 'fascinating'])} narrative that will keep you hooked until the last page.",
        
        f"'{title}' is a {random.choice(['brilliant', 'masterful', 'stunning'])} {genre.lower()} work that "
        f"{random.choice(['showcases', 'highlights', 'demonstrates'])} {author}'s exceptional storytelling abilities."
    ]
    return random.choice(templates)

def generate_audiobooks(num_books: int) -> pd.DataFrame:
    """Generate mock audiobook data."""
    audiobooks = []
    
    for book_id in range(1, num_books + 1):
        # Generate title
        title_words = [random.choice(ADJECTIVES), random.choice(NOUNS)]
        if random.random() > 0.5:  # Sometimes add an extra word
            title_words.append(random.choice(NOUNS))
        title = " ".join(title_words)
        
        # Generate author name
        author = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        # Generate narrator name
        narrator = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        
        # Generate other fields
        genre = random.choice(GENRES)
        book_tags = random.sample(TAGS, random.randint(2, 5))
        tags = ", ".join(book_tags)
        
        # Generate duration (in minutes, between 3-20 hours)
        duration = random.randint(180, 1200)
        
        # Generate description
        description = generate_description(title, author, genre, book_tags)
        
        audiobooks.append({
            'book_id': book_id,
            'title': title,
            'author': author,
            'narrator': narrator,
            'genre': genre,
            'duration': duration,
            'description': description,
            'tags': tags,
            'rating': round(random.uniform(3.0, 5.0), 1)
        })
    
    return pd.DataFrame(audiobooks)

def generate_user_interactions(num_users: int, audiobooks_df: pd.DataFrame) -> pd.DataFrame:
    """Generate mock user interaction data."""
    interactions = []
    
    for user_id in range(1, num_users + 1):
        # Each user has listened to between 5 and 30 books
        num_books = random.randint(5, 30)
        selected_books = random.sample(list(audiobooks_df['book_id']), num_books)
        
        # Generate interactions for each book
        for book_id in selected_books:
            # Progress as percentage (some books not finished)
            progress = random.randint(10, 100)
            
            # Generate realistic timestamps within the last year
            start_date = datetime.now() - timedelta(days=365)
            random_days = random.randint(0, 365)
            timestamp = start_date + timedelta(days=random_days)
            
            interactions.append({
                'user_id': user_id,
                'book_id': book_id,
                'progress': progress,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'rating': round(random.uniform(1.0, 5.0), 1) if random.random() > 0.3 else None
            })
    
    return pd.DataFrame(interactions)

def main():
    """Generate and save mock datasets."""
    # Generate audiobooks
    print("Generating audiobook data...")
    audiobooks_df = generate_audiobooks(200)
    audiobooks_df.to_csv('data/raw/audiobooks.csv', index=False)
    print(f"Generated {len(audiobooks_df)} audiobooks")
    
    # Generate user interactions
    print("Generating user interaction data...")
    interactions_df = generate_user_interactions(100, audiobooks_df)
    interactions_df.to_csv('data/raw/user_interactions.csv', index=False)
    print(f"Generated {len(interactions_df)} user interactions")

if __name__ == "__main__":
    main() 