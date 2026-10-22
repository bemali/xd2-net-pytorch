import sys
from pathlib import Path

# Make the repo importable (xd2net, tests.synthetic) without installing it.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
