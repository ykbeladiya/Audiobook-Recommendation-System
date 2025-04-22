import os
import json
import subprocess
from dotenv import load_dotenv

def deploy():
    # Load environment variables
    load_dotenv()
    
    # Get AWS account ID
    aws_account_id = os.getenv('AWS_ACCOUNT_ID')
    if not aws_account_id:
        raise ValueError("AWS_ACCOUNT_ID not found in .env file")
    
    # Update zappa_settings.json with actual AWS account ID
    with open('zappa_settings.json', 'r') as f:
        settings = json.load(f)
    
    # Replace placeholders with actual values
    for env in ['dev', 'prod']:
        settings[env]['s3_bucket'] = settings[env]['s3_bucket'].replace('${AWS_ACCOUNT_ID}', aws_account_id)
        settings[env]['role_arn'] = settings[env]['role_arn'].replace('${AWS_ACCOUNT_ID}', aws_account_id)
    
    # Write updated settings
    with open('zappa_settings.json', 'w') as f:
        json.dump(settings, f, indent=4)
    
    # Deploy to specified environment
    env = os.getenv('STAGE', 'dev')
    print(f"Deploying to {env} environment...")
    
    try:
        # Check if already deployed
        result = subprocess.run(['zappa', 'status', env], capture_output=True, text=True)
        if "not found" in result.stderr:
            # Initial deployment
            subprocess.run(['zappa', 'deploy', env], check=True)
        else:
            # Update existing deployment
            subprocess.run(['zappa', 'update', env], check=True)
        
        # Get the API URL
        status = subprocess.run(['zappa', 'status', env], capture_output=True, text=True)
        print("\nDeployment successful!")
        print("API URL can be found in the output above")
        
    except subprocess.CalledProcessError as e:
        print(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    deploy() 