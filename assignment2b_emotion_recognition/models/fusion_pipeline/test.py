"""Test the fusion pipeline. See common/pipeline.py for the arguments, e.g.

    python models/fusion_pipeline/test.py --features features --run Results/runs/fusion/word/seed0
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from common.pipeline import test_main  # noqa: E402

if __name__ == "__main__":
    test_main("fusion")
