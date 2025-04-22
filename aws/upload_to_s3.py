import os
import boto3
from botocore.exceptions import ClientError
import logging
from pathlib import Path
from dotenv import load_dotenv
import threading
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressPercentage:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                f"\r{self._filename}: {percentage:.2f}% ({self._seen_so_far}/{self._size} bytes)"
            )
            sys.stdout.flush()

class S3Uploader:
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
    
    def create_buckets(self):
        """Create S3 buckets if they don't exist."""
        for bucket_name in [self.data_bucket, self.model_bucket]:
            try:
                self.s3_client.head_bucket(Bucket=bucket_name)
                logger.info(f"Bucket {bucket_name} already exists")
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == '404':
                    try:
                        if self.region == 'us-east-1':
                            self.s3_client.create_bucket(Bucket=bucket_name)
                        else:
                            self.s3_client.create_bucket(
                                Bucket=bucket_name,
                                CreateBucketConfiguration={'LocationConstraint': self.region}
                            )
                        # Enable versioning
                        self.s3_client.put_bucket_versioning(
                            Bucket=bucket_name,
                            VersioningConfiguration={'Status': 'Enabled'}
                        )
                        logger.info(f"Created bucket {bucket_name} with versioning enabled")
                    except ClientError as e:
                        logger.error(f"Error creating bucket {bucket_name}: {str(e)}")
                        raise
                else:
                    logger.error(f"Error checking bucket {bucket_name}: {str(e)}")
                    raise
    
    def upload_file(self, file_path: str, bucket_name: str, s3_key: str = None):
        """
        Upload a file to S3 bucket with progress tracking.
        
        Args:
            file_path: Local path to the file
            bucket_name: Name of the S3 bucket
            s3_key: The key (path) where the file will be stored in S3
        """
        if s3_key is None:
            s3_key = os.path.basename(file_path)
            
        try:
            logger.info(f"Uploading {file_path} to {bucket_name}/{s3_key}")
            self.s3_client.upload_file(
                file_path, 
                bucket_name, 
                s3_key,
                Callback=ProgressPercentage(file_path)
            )
            print()  # New line after progress bar
            logger.info(f"Successfully uploaded {file_path}")
        except ClientError as e:
            logger.error(f"Error uploading {file_path}: {str(e)}")
            raise
    
    def upload_directory(self, local_dir: str, bucket_name: str, prefix: str = ""):
        """
        Upload all files in a directory to S3 bucket.
        
        Args:
            local_dir: Local directory path
            bucket_name: Name of the S3 bucket
            prefix: Prefix to add to S3 keys
        """
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Create S3 key maintaining directory structure
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
                
                # Skip temporary and system files
                if file.startswith('.') or file.endswith(('.pyc', '.pyo', '.pyd')):
                    continue
                
                self.upload_file(local_path, bucket_name, s3_key)

def main():
    """Main function to upload data and models to S3."""
    try:
        uploader = S3Uploader()
        
        # Create buckets if they don't exist
        logger.info("Creating/verifying S3 buckets...")
        uploader.create_buckets()
        
        # Get project root directory
        project_root = Path(__file__).parent.parent
        
        # Upload data files
        data_dir = project_root / "data"
        if data_dir.exists():
            logger.info("Uploading data files...")
            # Upload raw data
            raw_data_dir = data_dir / "raw"
            if raw_data_dir.exists():
                uploader.upload_directory(
                    str(raw_data_dir),
                    uploader.data_bucket,
                    "raw"
                )
            
            # Upload processed data
            processed_data_dir = data_dir / "processed"
            if processed_data_dir.exists():
                uploader.upload_directory(
                    str(processed_data_dir),
                    uploader.data_bucket,
                    "processed"
                )
        
        # Upload model files if they exist
        models_dir = project_root / "models"
        if models_dir.exists():
            logger.info("Uploading model files...")
            uploader.upload_directory(
                str(models_dir),
                uploader.model_bucket,
                "trained_models"
            )
        
        logger.info("Upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during upload process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 