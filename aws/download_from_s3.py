import os
import boto3
from botocore.exceptions import ClientError
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3Downloader:
    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        load_dotenv()
        
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.region = os.getenv('AWS_REGION', 'us-east-1')
        self.data_bucket = os.getenv('DATA_BUCKET')
        self.model_bucket = os.getenv('MODEL_BUCKET')
        
        if not all([self.aws_access_key, self.aws_secret_key, self.data_bucket, self.model_bucket]):
            raise ValueError("Missing required AWS credentials or bucket names in .env file")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key,
            region_name=self.region
        )
    
    def download_file(self, bucket_name: str, s3_key: str, local_path: str):
        """
        Download a file from S3 bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            s3_key: The key (path) of the file in S3
            local_path: Local path where the file will be saved
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(bucket_name, s3_key, local_path)
            logger.info(f"Successfully downloaded {s3_key} to {local_path}")
        except ClientError as e:
            logger.error(f"Error downloading {s3_key}: {str(e)}")
            raise
    
    def download_directory(self, bucket_name: str, prefix: str, local_dir: str):
        """
        Download all files from a directory in S3 bucket.
        
        Args:
            bucket_name: Name of the S3 bucket
            prefix: Prefix (directory) in S3 to download from
            local_dir: Local directory to save files to
        """
        try:
            # List objects in the bucket with the given prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    # Skip if the object is a directory
                    if s3_key.endswith('/'):
                        continue
                        
                    # Create local path
                    relative_path = s3_key[len(prefix):].lstrip('/')
                    local_path = os.path.join(local_dir, relative_path)
                    
                    # Download the file
                    self.download_file(bucket_name, s3_key, local_path)
                    
        except ClientError as e:
            logger.error(f"Error downloading directory {prefix}: {str(e)}")
            raise

def main():
    """Main function to download data and models from S3."""
    try:
        downloader = S3Downloader()
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Download data files
        data_dir = project_root / "data"
        logger.info("Downloading data files...")
        downloader.download_directory(
            downloader.data_bucket,
            "raw/",
            str(data_dir / "raw")
        )
        
        # Download model files
        models_dir = project_root / "models"
        logger.info("Downloading model files...")
        downloader.download_directory(
            downloader.model_bucket,
            "trained_models/",
            str(models_dir)
        )
        
        logger.info("Download completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during download process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 