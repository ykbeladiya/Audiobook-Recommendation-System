import subprocess
import time
import sys
import os

def main():
    # Start the FastAPI server
    print("\n🚀 Starting FastAPI server...")
    server_process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    try:
        # Run the test script
        print("\n🧪 Running API tests...")
        test_process = subprocess.run(
            [sys.executable, "api/test_api.py"],
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed with error: {str(e)}")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    finally:
        # Cleanup: Stop the server
        print("\n🛑 Stopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    main() 