
import torch
import yaml
from model.mutaplm import MutaPLM

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level (e.g., 'INFO', 'DEBUG')

    Returns:
        Configured logger instance
    """
    # Create output directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_embeddings_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from protein sequences"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., 'INFO', 'DEBUG')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(Path(args.log_dir), args.log_level)
    
    # load model
    device = torch.device("cuda")
    model_config_path = "../configs/mutaplm_inference.yaml"
    model_cfg = yaml.load(open(model_config_path, "r"), Loader=yaml.Loader)
    model_cfg["device"] = device
    model = MutaPLM(**model_cfg).to(device)    

if __name__ == "__main__":
    import os
    print(os.getcwd())
    main()