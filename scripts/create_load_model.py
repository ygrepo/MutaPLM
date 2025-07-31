# scripts/create_load_model.py
import sys
from pathlib import Path

# Make imports robust regardless of CWD (repo layout: <repo>/{model,scripts,configs,...})
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from model.model_util import select_device, create_model, load_model_safely  , load_model
import argparse
import logging
from datetime import datetime

def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"extract_embeddings_{ts}.log"
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")
    return logger


def parse_args():
    p = argparse.ArgumentParser(description="Extract embeddings from protein sequences")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "mutaplm_inference.yaml"))
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N|mps")
    p.add_argument("--checkpoint_path", type=str, default=str(REPO_ROOT / "ckpts" / "mutaplm.pth"))
    return p.parse_args()


# def test_fused_embeddings(model):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     wildtype_protein = "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"
#     site = "A70K"
#     mutated_protein = wildtype_protein[:int(site[1:-1])-1] + site[-1] + wildtype_protein[int(site[1:-1]):]
#     (vec, delta) = get_fused(model, wildtype_protein, mutated_protein, site)
#     logger.info(f"Fused embeddings: {vec.shape}")
#     logger.info(f"Mutation delta: {delta.shape}") 

# def test_fused_soft_embeddings(model):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     wildtype_protein = "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"
#     site = "A70K"
#     mutated_protein = wildtype_protein[:int(site[1:-1])-1] + site[-1] + wildtype_protein[int(site[1:-1]):]
#     (vec_soft, soft_only) = fused_soft(model, wildtype_protein, mutated_protein, site)
#     logger.info(f"Fused soft embeddings: {vec_soft.shape}")
#     logger.info(f"Soft only: {soft_only.shape}") 


def main():

    args = parse_args()
    logger = setup_logging(Path(args.log_dir), args.log_level)
    device = select_device(args.device)
    logger.info(f"Using device: {device}")
    model = create_model(Path(args.config), device)
    #load_model_safely(model, args.checkpoint_path, device=device, weights_only=False, strict=False)
    model = load_model(model, args.checkpoint_path)
    logger.info("Model loaded successfully.")
    # check(model)

    #test_fused_embeddings(model)
    #test_fused_soft_embeddings(model)
    
if __name__ == "__main__":
    main()
