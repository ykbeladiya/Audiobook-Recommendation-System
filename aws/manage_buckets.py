import os
import boto3
from botocore.exceptions import ClientError
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3BucketManager:
    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        load_dotenv()
        
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not all([self.aws_access_key, self.aws_secret_key]):
            raise ValueError("Missing required AWS credentials in .env file")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
    
    def create_bucket(self, bucket_name: str):
        """
        Create an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket to create
        """
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            logger.info(f"Successfully created bucket: {bucket_name}")
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            logger.info(f"Enabled versioning for bucket: {bucket_name}")
            
        except ClientError as e:
            logger.error(f"Error creating bucket {bucket_name}: {str(e)}")
            raise
    
    def list_buckets(self):
        """List all S3 buckets in the account."""
        try:
            response = self.s3_client.list_buckets()
            buckets = response['Buckets']
            
            logger.info("Available buckets:")
            for bucket in buckets:
                logger.info(f"- {bucket['Name']} (Created: {bucket['CreationDate']})")
                
            return buckets
            
        except ClientError as e:
            logger.error(f"Error listing buckets: {str(e)}")
            raise
    
    def delete_bucket(self, bucket_name: str, force: bool = False):
        """
        Delete an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket to delete
            force: If True, delete all objects in the bucket before deleting the bucket
        """
        try:
            if force:
                # Delete all objects and versions
                paginator = self.s3_client.get_paginator('list_object_versions')
                for page in paginator.paginate(Bucket=bucket_name):
                    versions = page.get('Versions', [])
                    delete_markers = page.get('DeleteMarkers', [])
                    
                    # Delete objects
                    if versions or delete_markers:
                        objects_to_delete = [
                            {'Key': version['Key'], 'VersionId': version['VersionId']}
                            for version in versions + delete_markers
                        ]
                        
                        self.s3_client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
            
            # Delete the bucket
            self.s3_client.delete_bucket(Bucket=bucket_name)
            logger.info(f"Successfully deleted bucket: {bucket_name}")
            
        except ClientError as e:
            logger.error(f"Error deleting bucket {bucket_name}: {str(e)}")
            raise

def main():
    """Main function to manage S3 buckets."""
    try:
        manager = S3BucketManager()
        
        # Get bucket names from environment variables
        data_bucket = os.getenv('DATA_BUCKET')
        model_bucket = os.getenv('MODEL_BUCKET')
        
        if not all([data_bucket, model_bucket]):
            raise ValueError("Missing bucket names in .env file")
        
        # List existing buckets
        logger.info("Listing existing buckets...")
        existing_buckets = manager.list_buckets()
        existing_bucket_names = [b['Name'] for b in existing_buckets]
        
        # Create buckets if they don't exist
        for bucket_name in [data_bucket, model_bucket]:
            if bucket_name not in existing_bucket_names:
                logger.info(f"Creating bucket {bucket_name}...")
                manager.create_bucket(bucket_name)
        
        logger.info("Bucket management completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during bucket management: {str(e)}")
        raise

if __name__ == "__main__":
    main() 