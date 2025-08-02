import argparse
import logging
logger = logging.getLogger(__name__)

from trainer import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Trainer.add_arguments(parser)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()