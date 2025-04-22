import os
import boto3
from botocore.exceptions import ClientError
import logging
from decimal import Decimal
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamoDBQuerier:
    def __init__(self):
        """Initialize DynamoDB client with credentials from environment variables."""
        load_dotenv()
        
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not all([self.aws_access_key, self.aws_secret_key]):
            raise ValueError("Missing required AWS credentials in .env file")
        
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
        
        # Table names from environment variables
        self.users_table_name = os.getenv('USER_TABLE', 'audiobook-users')
        self.books_table_name = os.getenv('BOOK_TABLE', 'audiobook-books')
        self.interactions_table_name = os.getenv('INTERACTION_TABLE', 'audiobook-interactions')
        
        # Get table references
        self.users_table = self.dynamodb.Table(self.users_table_name)
        self.books_table = self.dynamodb.Table(self.books_table_name)
        self.interactions_table = self.dynamodb.Table(self.interactions_table_name)
    
    @staticmethod
    def decimal_to_float(obj: Any) -> Any:
        """Convert Decimal objects to float for JSON serialization."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: DynamoDBQuerier.decimal_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DynamoDBQuerier.decimal_to_float(v) for v in obj]
        return obj
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        try:
            response = self.users_table.get_item(Key={'user_id': user_id})
            item = response.get('Item')
            if item:
                return self.decimal_to_float(item)
            return None
        except ClientError as e:
            logger.error(f"Error getting user {user_id}: {str(e)}")
            raise
    
    def get_book(self, book_id: int) -> Optional[Dict[str, Any]]:
        """Get book by ID."""
        try:
            response = self.books_table.get_item(Key={'book_id': book_id})
            item = response.get('Item')
            if item:
                return self.decimal_to_float(item)
            return None
        except ClientError as e:
            logger.error(f"Error getting book {book_id}: {str(e)}")
            raise
    
    def get_user_interactions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all interactions for a user."""
        try:
            response = self.interactions_table.query(
                KeyConditionExpression='user_id = :uid',
                ExpressionAttributeValues={':uid': user_id}
            )
            items = response.get('Items', [])
            return self.decimal_to_float(items)
        except ClientError as e:
            logger.error(f"Error getting interactions for user {user_id}: {str(e)}")
            raise
    
    def get_books_by_genre(self, genre: str) -> List[Dict[str, Any]]:
        """Get all books in a specific genre using the GenreIndex."""
        try:
            response = self.books_table.query(
                IndexName='GenreIndex',
                KeyConditionExpression='genre = :g',
                ExpressionAttributeValues={':g': genre}
            )
            items = response.get('Items', [])
            return self.decimal_to_float(items)
        except ClientError as e:
            logger.error(f"Error getting books for genre {genre}: {str(e)}")
            raise
    
    def get_user_recent_interactions(self, user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's most recent interactions using the TimestampIndex."""
        try:
            response = self.interactions_table.query(
                IndexName='TimestampIndex',
                KeyConditionExpression='user_id = :uid',
                ExpressionAttributeValues={':uid': user_id},
                ScanIndexForward=False,  # Sort in descending order
                Limit=limit
            )
            items = response.get('Items', [])
            return self.decimal_to_float(items)
        except ClientError as e:
            logger.error(f"Error getting recent interactions for user {user_id}: {str(e)}")
            raise
    
    def update_user(self, user_id: int, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user attributes."""
        try:
            update_expression = "SET " + ", ".join(f"#{k} = :{k}" for k in updates.keys())
            expression_values = {f":{k}": v for k, v in updates.items()}
            expression_names = {f"#{k}": k for k in updates.keys()}
            
            response = self.users_table.update_item(
                Key={'user_id': user_id},
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
                ExpressionAttributeNames=expression_names,
                ReturnValues="ALL_NEW"
            )
            return self.decimal_to_float(response.get('Attributes', {}))
        except ClientError as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            raise
    
    def delete_user(self, user_id: int):
        """Delete a user and all their interactions."""
        try:
            # Delete user's interactions first
            interactions = self.get_user_interactions(user_id)
            with self.interactions_table.batch_writer() as batch:
                for interaction in interactions:
                    batch.delete_item(
                        Key={
                            'user_id': user_id,
                            'book_id': interaction['book_id']
                        }
                    )
            
            # Delete user
            self.users_table.delete_item(Key={'user_id': user_id})
            logger.info(f"Successfully deleted user {user_id} and their interactions")
            
        except ClientError as e:
            logger.error(f"Error deleting user {user_id}: {str(e)}")
            raise

def main():
    """Example usage of the DynamoDB querier."""
    try:
        querier = DynamoDBQuerier()
        
        # Example queries
        user_id = 1
        book_id = 1
        genre = "Fiction"
        
        # Get user details
        user = querier.get_user(user_id)
        if user:
            logger.info(f"User {user_id} details: {json.dumps(user, indent=2)}")
        
        # Get book details
        book = querier.get_book(book_id)
        if book:
            logger.info(f"Book {book_id} details: {json.dumps(book, indent=2)}")
        
        # Get books by genre
        genre_books = querier.get_books_by_genre(genre)
        logger.info(f"Found {len(genre_books)} books in genre {genre}")
        
        # Get user's recent interactions
        recent_interactions = querier.get_user_recent_interactions(user_id, limit=5)
        logger.info(f"User {user_id}'s recent interactions: {json.dumps(recent_interactions, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error during queries: {str(e)}")
        raise

if __name__ == "__main__":
    main() 