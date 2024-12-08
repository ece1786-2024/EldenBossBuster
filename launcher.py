import subprocess

# Define the path to your backend script
BACKEND_SCRIPT = "app.py"

def start_backend():
    """Start the Flask backend."""
    print("Starting the backend server...")
    return subprocess.Popen(["python", BACKEND_SCRIPT])

if __name__ == "__main__":
    # Start the backend process
    backend_process = start_backend()

    try:
        # Wait for the backend process to complete
        backend_process.wait()
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully
        print("Shutting down...")
        backend_process.terminate()
        print("Backend server has been stopped.")
