"""
Master pipeline runner — executes all phases in sequence.

Usage:
    python run_pipeline.py            # all phases
    python run_pipeline.py --phase 2  # specific phase only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def phase2():
    from src.features import run_feature_pipeline
    return run_feature_pipeline()


def phase3():
    from src.models import run_training
    return run_training()


def phase4():
    from src.evaluate import run_evaluation
    return run_evaluation()


def main():
    parser = argparse.ArgumentParser(description="LTE Handover Prediction Pipeline")
    parser.add_argument("--phase", type=int, choices=[2, 3, 4],
                        help="Run a single phase (2=features, 3=train, 4=eval)")
    args = parser.parse_args()

    if args.phase == 2:
        phase2()
    elif args.phase == 3:
        phase3()
    elif args.phase == 4:
        phase4()
    else:
        print("=" * 55)
        print("  LTE Handover Prediction — Full Pipeline")
        print("=" * 55)
        phase2()
        phase3()
        phase4()
        print("\nAll phases complete.")
        print("Launch dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
