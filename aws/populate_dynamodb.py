import os
import boto3
import pandas as pd
from dotenv import load_dotenv
import logging
from pathlib import Path
from decimal import Decimal
import json
from typing import List, Dict, Any
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamoDBPopulator:
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
        
        self.table_schemas = {
            'users': {
                'KeySchema': [
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'user_id', 'AttributeType': 'S'}
                ],
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            'books': {
                'KeySchema': [
                    {'AttributeName': 'book_id', 'KeyType': 'HASH'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'book_id', 'AttributeType': 'S'}
                ],
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            },
            'user_interactions': {
                'KeySchema': [
                    {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'book_id', 'KeyType': 'RANGE'}
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'user_id', 'AttributeType': 'S'},
                    {'AttributeName': 'book_id', 'AttributeType': 'S'}
                ],
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5
                }
            }
        }

    def create_tables(self):
        """Create DynamoDB tables if they don't exist."""
        for table_name, schema in self.table_schemas.items():
            try:
                table = self.dynamodb.create_table(
                    TableName=table_name,
                    **schema
                )
                logger.info(f"Creating table {table_name}...")
                table.meta.client.get_waiter('table_exists').wait(TableName=table_name)
                logger.info(f"Table {table_name} created successfully")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceInUseException':
                    logger.info(f"Table {table_name} already exists")
                else:
                    logger.error(f"Error creating table {table_name}: {str(e)}")
                    raise

    @staticmethod
    def convert_to_dynamodb_format(item: Dict[str, Any]) -> Dict[str, Any]:
        """Convert item values to DynamoDB-compatible format."""
        for key, value in item.items():
            if pd.isna(value):
                item[key] = None
            elif isinstance(value, (float, int)):
                item[key] = Decimal(str(value))
            elif isinstance(value, (list, dict)):
                item[key] = json.dumps(value)
        return item

    def batch_write_items(self, table_name: str, items: List[Dict[str, Any]]):
        """Write items to DynamoDB table in batches."""
        table = self.dynamodb.Table(table_name)
        with table.batch_writer() as batch:
            for item in items:
                try:
                    dynamodb_item = self.convert_to_dynamodb_format(item)
                    batch.put_item(Item=dynamodb_item)
                except ClientError as e:
                    logger.error(f"Error writing item to {table_name}: {str(e)}")
                    raise

    def populate_users_table(self, ratings_file: str):
        """Extract unique users from ratings and populate users table."""
        df = pd.read_csv(ratings_file)
        unique_users = df[['user_id']].drop_duplicates()
        users = [{'user_id': str(user_id)} for user_id in unique_users['user_id']]
        
        logger.info(f"Populating users table with {len(users)} users...")
        self.batch_write_items('users', users)
        logger.info("Users table populated successfully")

    def populate_books_table(self, books_file: str):
        """Populate books table from books.csv."""
        df = pd.read_csv(books_file)
        books = df.to_dict('records')
        books = [{k: str(v) if k == 'book_id' else v for k, v in book.items()} 
                for book in books]
        
        logger.info(f"Populating books table with {len(books)} books...")
        self.batch_write_items('books', books)
        logger.info("Books table populated successfully")

    def populate_interactions_table(self, interactions_file: str):
        """Populate user_interactions table from user_interactions.csv."""
        df = pd.read_csv(interactions_file)
        interactions = df.to_dict('records')
        interactions = [{k: str(v) if k in ['user_id', 'book_id'] else v 
                        for k, v in interaction.items()} 
                       for interaction in interactions]
        
        logger.info(f"Populating user_interactions table with {len(interactions)} interactions...")
        self.batch_write_items('user_interactions', interactions)
        logger.info("User interactions table populated successfully")

def main():
    """Main function to populate DynamoDB tables."""
    try:
        populator = DynamoDBPopulator()
        
        # Create tables if they don't exist
        logger.info("Creating/verifying DynamoDB tables...")
        populator.create_tables()
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "raw"
        
        # Populate tables
        populator.populate_users_table(str(data_dir / "user_ratings.csv"))
        populator.populate_books_table(str(data_dir / "books.csv"))
        populator.populate_interactions_table(str(data_dir / "user_interactions.csv"))
        
        logger.info("All tables populated successfully!")
        
    except Exception as e:
        logger.error(f"Error during population process: {str(e)}")
        raise

if __name__ == "__main__":
    main()