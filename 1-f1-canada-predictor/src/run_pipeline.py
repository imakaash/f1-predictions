"""End-to-end pipeline: fetch -> features -> train -> predict.

Run:
    python -m src.run_pipeline
"""
from src import build_features, fetch_data, predict, train


def main() -> None:
    fetch_data.main()
    build_features.main()
    train.main()
    predict.main()


if __name__ == "__main__":
    main()
