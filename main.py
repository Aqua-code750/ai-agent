import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(Path(__file__).parent / "my_agent" / ".env")

# Add current directory and my_agent to sys.path
root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "my_agent"))

from my_agent.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
