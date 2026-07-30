from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> None:
    print("AniMedika environment check")
    print("=" * 40)
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"Operating system: {platform.platform()}")

    try:
        import tensorflow as tf

        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {getattr(tf.keras, '__version__', 'bundled with TensorFlow')}")
        print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")
    except Exception as error:
        print(f"TensorFlow import failed: {error}")
        raise

    try:
        import numpy as np
        import pandas as pd
        import sklearn
        import PIL

        print(f"NumPy version: {np.__version__}")
        print(f"pandas version: {pd.__version__}")
        print(f"scikit-learn version: {sklearn.__version__}")
        print(f"Pillow version: {PIL.__version__}")
    except Exception as error:
        print(f"Dependency import failed: {error}")
        raise

    project_root = Path(__file__).resolve().parent.parent
    print(f"Project root: {project_root}")
    print(f"Dataset folder: {project_root / 'dataset'}")
    print(f"Models folder: {project_root / 'models'}")
    print(f"Outputs folder: {project_root / 'outputs'}")
    print("\nEnvironment check passed.")


if __name__ == "__main__":
    main()
