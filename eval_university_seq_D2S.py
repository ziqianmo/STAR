"""Evaluate STAR on university: sequence queries."""

from dataclasses import dataclass

from star.evaluate.cli import EvaluationConfig, run_evaluation


@dataclass
class Configuration(EvaluationConfig):
    dataset_name: str = "university"
    sequence_query: bool = True


# Edit EvaluationConfig fields here, e.g. config.checkpoint_start below.
config = Configuration()


def main(argv=None):
    return run_evaluation(config, argv)


if __name__ == "__main__":
    main()
