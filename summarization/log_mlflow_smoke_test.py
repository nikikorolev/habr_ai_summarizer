from __future__ import annotations

import pandas as pd

from mlflow_logging import log_summarization_results


def main() -> None:
    results = pd.DataFrame(
        [
            {
                "model": "ru-gazeta",
                "type": "generative",
                "rouge_l": 0.3476,
                "meteor": 0.1955,
                "keyword_coverage": 0.3121,
                "compression_ratio": 0.0694,
                "overall_score": 0.8714,
            },
            {
                "model": "e5-ru-gazeta-top-30",
                "type": "extractive",
                "rouge_l": 0.3231,
                "meteor": 0.1954,
                "keyword_coverage": 0.2712,
                "compression_ratio": 0.0688,
                "overall_score": 0.7553,
            },
        ]
    ).set_index("model")

    run_id = log_summarization_results(
        results,
        run_prefix="smoke-test",
        sample_size=50,
    )
    print(f"Logged smoke-test MLflow run: {run_id}")


if __name__ == "__main__":
    main()
