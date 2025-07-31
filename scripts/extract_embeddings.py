# scripts/extract_embeddings.py
import sys
from pathlib import Path

# Make imports robust regardless of CWD (repo layout: <repo>/{model,scripts,configs,...})
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import yaml
import torch
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

def select_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref.startswith("cuda"):
        return torch.device(pref) if torch.cuda.is_available() else torch.device("cpu")
    if pref == "mps":
        return torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def create_model(cfg_path: Path, device):
    from model.mutaplm import MutaPLM  # after sys.path insert
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open() as f:
        model_cfg = yaml.safe_load(f)

    model_cfg["device"] = device
    model = MutaPLM(**model_cfg).to(device).eval()

    # Keep CPU in float32 (your class defaults to bf16 for from_pretrained)
    if device.type != "cuda":
        model.float()

    logger.info("Model loaded successfully.")

def load_model(model, checkpoint_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"Loading model checkpoint from {checkpoint_path}")
    new_ckpt = torch.load(open(checkpoint_path, "rb"), map_location="cuda")["model"]
    logger.info("Model checkpoint loaded successfully.")
    logger.info("Loading model state dict...")
    model.load_state_dict(new_ckpt, strict=False)
    logger.info("Model state dict loaded successfully.")

    model.eval()

def check(model):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Should NOT be near-xavier random std; vocab embeddings should have non-trivial stats
    emb = next(model.llm.parameters()).detach()
    logger.info(emb.mean().item(), emb.std().item())
    
    for n in ["proj_protein1.weight", "query_protein1", "soft_tokens"]:
        p = dict(model.named_parameters())[n].detach()
        logger.info(n, p.mean().item(), p.std().item())

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
    load_model(model, args.checkpoint_path)
    check(model)

    #test_fused_embeddings(model)
    #test_fused_soft_embeddings(model)
    
if __name__ == "__main__":
    #print("CWD:", os.getcwd())
    main()
