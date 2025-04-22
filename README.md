# Audiobook Recommendation System 🎧

A sophisticated recommendation system for audiobooks that combines collaborative filtering and content-based approaches to provide personalized book recommendations to users.

## Features ✨

- **Hybrid Recommendation Engine** (60% collaborative, 40% content-based)
  - User-based collaborative filtering
  - Content-based recommendations using book metadata
  - Hybrid recommendations combining both approaches

- **FastAPI Backend**
  - RESTful API endpoints
  - Swagger documentation
  - Efficient data processing
  - CORS support

- **Streamlit Frontend**
  - User-friendly interface
  - Real-time recommendations
  - Book details display
  - Multiple recommendation types

## Installation 🚀

### Prerequisites

- Python 3.8+
- PostgreSQL
- Node.js (for AWS deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Audiobook-Recommendation-System.git
cd Audiobook-Recommendation-System
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up PostgreSQL:
```bash
# Install PostgreSQL using Chocolatey (Windows)
choco install postgresql -y

# Create database and user (run in psql)
CREATE DATABASE audiobook_recommendations;
CREATE USER audiobook_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE audiobook_recommendations TO audiobook_user;
```

5. Create a `.env` file:
```env
DATABASE_URL=postgresql://audiobook_user:your_password@localhost:5432/audiobook_recommendations
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
S3_BUCKET_NAME=your-bucket-name
```

## Running Locally 🏃‍♂️

1. Start the FastAPI server:
```bash
uvicorn src.main:app --reload
```

2. In a new terminal, start the Streamlit app:
```bash
streamlit run app.py
```

3. Access the applications:
   - FastAPI Swagger docs: http://localhost:8000/docs
   - Streamlit interface: http://localhost:8501

## AWS Deployment with Zappa 🚀

1. Configure AWS credentials:
```bash
aws configure
```

2. Initialize Zappa:
```bash
zappa init
```

3. Deploy to AWS Lambda:
```bash
zappa deploy production
```

4. Update deployment (if needed):
```bash
zappa update production
```

## API Documentation 📚

### Endpoints

1. User-Based Recommendations
```bash
GET /recommendations/user/{user_id}
```

2. Similar Books
```bash
GET /recommendations/similar/{book_id}
```

3. Hybrid Recommendations
```bash
GET /recommendations/hybrid/{user_id}
```

### Example API Call

```python
import requests

# Get hybrid recommendations for user 1
response = requests.get(
    "http://localhost:8000/recommendations/hybrid/1",
    params={"limit": 5}
)
recommendations = response.json()
```


## Architecture 🏗️

```
Audiobook-Recommendation-System/
├── src/
│   ├── models/
│   │   ├── collaborative_filtering.py
│   │   └── content_based.py
│   ├── utils/
│   │   └── logger.py
│   ├── main.py
│   └── recommendation_engine.py
├── data/
│   └── raw/
│       ├── books.csv
│       └── user_ratings.csv
├── app.py
├── requirements.txt
└── README.md
```

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- scikit-learn for machine learning algorithms
- FastAPI for the backend framework
- Streamlit for the user interface
- PostgreSQL for data storage
