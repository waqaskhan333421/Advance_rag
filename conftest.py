import sys
from pathlib import Path

# Add project root directory to sys.path for pytest discovery
sys.path.insert(0, str(Path(__file__).parent))
