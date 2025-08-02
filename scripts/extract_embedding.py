
import sys
from pathlib import Path
from tqdm import tqdm

# Make imports robust regardless of CWD (repo layout: <repo>/{model,scripts,configs,...})
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.model.model_util import (
    select_device, load_model_from_config, setup_logging, llm_context_cosine
)
import argparse
import pandas as pd
import logging
import torch
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    p = argparse.ArgumentParser(description="Create and load MutaPLM model")
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--log_level", type=str, default="INFO")
    p.add_argument("--config", type=str, default=str(REPO_ROOT / "configs" / "mutaplm_inference.yaml"))
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:N|mps")
    p.add_argument("--checkpoint_path", type=str, default=str(REPO_ROOT / "ckpts" / "mutaplm.pth"))
    p.add_argument("--data_fn", type=str, default="")
    p.add_argument("--output_fn", type=str, default="")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



# def test_embeddings(model, logger):
#     wildtype_protein = "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"
#     site = "A70K"
#     mutated_protein = wildtype_protein[:int(site[1:-1])-1] + site[-1] + wildtype_protein[int(site[1:-1]):]
#     v_wt, v_mut, cos = llm_context_cosine(model, wildtype_protein, mutated_protein)
#     logger.info(f"Cosine similarity: {cos}")
#     logger.info(f"WT embedding: {v_wt.shape}")
#     logger.info(f"Mut embedding: {v_mut.shape}")
   
# def test_fused_soft_embeddings(model):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     wildtype_protein = "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"
#     site = "A70K"
#     mutated_protein = wildtype_protein[:int(site[1:-1])-1] + site[-1] + wildtype_protein[int(site[1:-1]):]
#     (vec_soft, soft_only) = fused_soft(model, wildtype_protein, mutated_protein, site)
#     logger.info(f"Fused soft embeddings: {vec_soft.shape}")
#     logger.info(f"Soft only: {soft_only.shape}") 


@torch.no_grad()
def retrieve_embeddings(model, df, output_fn: Path, batch_size=16):
    protein1_embeddings = []
    protein2_embeddings = []
    cosine_sims = []

    # --- Process in batches ---
    model.eval()

    for start in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[start:start+batch_size]
        wt_list = batch_df["protein1"].tolist()
        mut_list = batch_df["protein2"].tolist()

        # Optional: move to GPU context for faster embeddings
        with model.maybe_autocast():
            for wt, mut in zip(wt_list, mut_list):
                v_wt, v_mut, cos = llm_context_cosine(model, wt, mut)
                protein1_embeddings.append(v_wt.cpu().numpy())
                protein2_embeddings.append(v_mut.cpu().numpy())
                cosine_sims.append(cos)

    # --- Combine results ---
    df["cosine_similarity"] = cosine_sims
    df["protein1_embedding"] = protein1_embeddings
    df["protein2_embedding"] = protein2_embeddings

    logger.info(f"Computed embeddings for {len(df)} protein pairs")
    logger.info(f"Example cosine similarity: {cosine_sims[:5]}")

    # Save embeddings if needed
    if output_fn:
        logger.info(f"Saving embeddings to {output_fn}")
        df.to_csv(output_fn, index=False)
        logger.info(f"Saved embeddings to {output_fn}")
        
def load_data(data_fn: Path, n: int, seed: int=42):
    df = pd.read_csv(data_fn, low_memory=False)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df = df.dropna(subset=["protein1", "protein2"])
    logger.info(f"Loaded dataset: {len(df)} rows")
    if n > 0:
        logger.info(f"Sampling {n} rows")
        df = df.sample(n=n, random_state=seed)
    return df

def main():
    args = parse_args()
    logger = setup_logging("extract_embedding", Path(args.log_dir), args.log_level)
    device = select_device(args.device)
    logger.info(f"Using device: {device}")
    model = load_model_from_config(device, Path(args.config), Path(args.checkpoint_path))
    logger.info("Model loaded successfully.")
    df = load_data(Path(args.data_fn), args.n, args.seed)
    retrieve_embeddings(model, df, Path(args.output_fn))
   
 
if __name__ == "__main__":     
    main()  