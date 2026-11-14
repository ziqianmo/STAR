"""Train STAR on sues200; edit config below or override it with CLI options."""

from dataclasses import dataclass

from star.training import TrainingConfig, run_training


@dataclass
class Configuration(TrainingConfig):
    dataset_name: str = "sues200"
    pretrained: bool = True
    lr: float = 0.0006
    nclasses: int = 200
    altitude: int = 300


# Edit any TrainingConfig field here; CLI options take precedence.
config = Configuration()


def main(argv=None):
    return run_training(config, argv, entry_file=__file__)


if __name__ == "__main__":
    main()
