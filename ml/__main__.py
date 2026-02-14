"""Allow running as `python -m ml` (in addition to `python -m ml.run_ml`)."""
from ml.run_ml import main
import sys
sys.exit(main())
