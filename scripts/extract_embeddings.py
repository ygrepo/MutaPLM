# scripts/extract_embeddings.py
import os
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

def create_model(cfg_path, device):
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

def check(model):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Should NOT be near-xavier random std; vocab embeddings should have non-trivial stats
    emb = next(model.llm.parameters()).detach()
    logger.info(emb.mean().item(), emb.std().item())
    
    for n in ["proj_protein1.weight", "query_protein1", "soft_tokens"]:
        p = dict(model.named_parameters())[n].detach()
        logger.info(n, p.mean().item(), p.std().item())

@torch.no_grad()
def fused_pre_llm(model, wt: str, mut: str):
    # ESM→LLM pooled tokens
    p1, p2 = model._encode_protein([wt], [mut])  # each: [1, Q, Hllm]
    p1_mean = p1.mean(dim=1)                     # [1, Hllm]
    p2_mean = p2.mean(dim=1)                     # [1, Hllm]
    delta    = p2_mean - p1_mean                 # [1, Hllm]
    # Simple fusions you can try:
    fused_cat   = torch.cat([p1_mean, p2_mean, delta], dim=-1)  # [1, 3*Hllm]
    return {
        "esm_llm_wt_tokens": p1, "esm_llm_mut_tokens": p2,
        "esm_llm_wt": p1_mean, "esm_llm_mut": p2_mean,
        "esm_llm_delta": delta, "fused": fused_cat
    }

@torch.no_grad()
def fused_in_llm(model, wt: str, mut: str, *,
                 func_text: str = "Describe the protein function.",
                 muta_prompt: str = "Describe the mutation impact."):
    """
    Returns a dict with contextualized LLM embeddings for WT & Mut and a fused vector.

    Preconditions:
      - `model` is a MutaPLM with a **pretrained LLM** loaded.
      - Ideally, the MutaPLM bridge (query tokens, projections, soft tokens) is finetuned.
      - Call model.eval() beforehand. On CPU, ensure model.float().

    Outputs:
      - "llm_ctx_wt"  : [1, H_llm]
      - "llm_ctx_mut" : [1, H_llm]
      - "llm_ctx_delta": [1, H_llm]
      - "esm_llm_wt", "esm_llm_mut": [1, H_llm] (pre-LLM pooled)
      - "fused"       : [1, 5*H_llm] (concat of above)
      - "spans"       : dict with (start, end) indices for P1/P2 in the LLM sequence
    """
    device = model.device if getattr(model, "device", None) is not None else next(model.parameters()).device

    # 1) ESM→LLM pooled tokens for WT & Mut
    #    p1, p2: [1, Q, H_llm] (already projected into LLM space)
    with model.maybe_autocast():  # safe on CUDA; nullcontext on CPU
        p1, p2 = model._encode_protein([wt], [mut])
    Q1, Q2 = p1.shape[1], p2.shape[1]

    # 2) Build the same wrapped inputs the FT path uses (we only need the first two outputs)
    #    NOTE: pass func_text without "</s>" (the method will append it internally)
    dummy_text = ["Short answer."]  # not used for pooling
    wrapped = model._wrapped_sentence_ft(
        protein1_embeds=p1,
        protein2_embeds=p2,
        mut_entry=["[1]"],          # dummy; only used if t2m=True, which we don't need here
        p_function=[func_text],     # DO NOT append "</s>" here; method does it
        muta_prompt=[muta_prompt],
        text=dummy_text
    )
    batched_embeds1, batched_attn_mask1 = wrapped[0], wrapped[1]  # [1, T, H_llm], [1, T]

    # 3) LLM forward to get contextual hidden states over the whole wrapped sequence
    with model.maybe_autocast():
        out = model.llm(
            inputs_embeds=batched_embeds1,
            attention_mask=batched_attn_mask1,
            output_hidden_states=True,
            return_dict=True
        ).hidden_states[-1]  # [1, T, H_llm]

    # 4) Compute exact spans for P1 and P2
    #     Use the exact SYS string hardcoded in _wrapped_sentence_ft of your class:
    sys_str = (
        "You are an expert at biology and life science. Now a user gives you several protein sequences "
        "and mutations. Please follow user instructions and answer their questions. Based on the following "
        "protein sequence, please describe its function."
    )
    # Token counts (no special tokens)
    sys_len  = len(model.llm_tokenizer(sys_str, add_special_tokens=False).input_ids)
    func_len = len(model.llm_tokenizer(func_text + "</s>", add_special_tokens=False).input_ids)  # they append "</s>"
    mut_len  = len(model.llm_tokenizer(muta_prompt, add_special_tokens=False).input_ids)

    # Layout:
    # [BOS(1), SYS(sys_len), BOP(1), P1(Q1), EOP(1), FUNC(func_len), MUT(mut_len), BOM(1), P2(Q2), EOM(1), TEXT(...)]
    idx = 0
    idx += 1                      # BOS
    idx += sys_len                # SYS
    idx += 1                      # BOP
    p1_start = idx
    p1_end   = p1_start + Q1
    idx = p1_end
    idx += 1                      # EOP
    idx += func_len               # FUNC
    idx += mut_len                # MUT
    idx += 1                      # BOM
    p2_start = idx
    p2_end   = p2_start + Q2
    # (we don't need to advance further for pooling)

    # 5) Pool the LLM hidden states across P1/P2 spans
    p1_ctx = out[:, p1_start:p1_end, :].mean(dim=1)  # [1, H_llm]
    p2_ctx = out[:, p2_start:p2_end, :].mean(dim=1)  # [1, H_llm]
    delta_ctx = p2_ctx - p1_ctx                      # [1, H_llm]

    # 6) (Optional) also include the pre-LLM pooled vectors
    p1_mean = p1.mean(dim=1)                         # [1, H_llm]
    p2_mean = p2.mean(dim=1)                         # [1, H_llm]

    fused = torch.cat([p1_mean, p2_mean, p1_ctx, p2_ctx, delta_ctx], dim=-1)  # [1, 5*H_llm]

    return {
        "llm_ctx_wt": p1_ctx, "llm_ctx_mut": p2_ctx, "llm_ctx_delta": delta_ctx,
        "esm_llm_wt": p1_mean, "esm_llm_mut": p2_mean,
        "fused": fused,
        "spans": {"p1": (p1_start, p1_end), "p2": (p2_start, p2_end)},
    }

def get_fused(model, wt, mut, site):
    #fusedA = fused_pre_llm(model, wt, mut)
    model.eval()
    if model.device.type != "cuda":
        model.float()  # keep CPU in fp32

    res = fused_in_llm(
        model,
        wt=wt,
        mut=mut,
        func_text="Summarize the protein's function.",
        muta_prompt=f"Mutation {site[0]}→{site[-1]} at position {site[1:-1]}."
    )

    vec = res["fused"]             # [1, 5*H_llm] fused embedding
    delta = res["llm_ctx_delta"]   # [1, H_llm] mutation delta (contextual)

    return vec, delta


def test_fused_embeddings(model):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    wildtype_protein = "MASDAAAEPSSGVTHPPRYVIGYALAPKKQQSFIQPSLVAQAASRGMDLVPVDASQPLAEQGPFHLLIHALYGDDWRAQLVAFAARHPAVPIVDPPHAIDRLHNRISMLQVVSELDHAADQDSTFGIPSQVVVYDAAALADFGLLAALRFPLIAKPLVADGTAKSHKMSLVYHREGLGKLRPPLVLQEFVNHGGVIFKVYVVGGHVTCVKRRSLPDVSPEDDASAQGSVSFSQVSNLPTERTAEEYYGEKSLEDAVVPPAAFINQIAGGLRRALGLQLFNFDMIRDVRAGDRYLVIDINYFPGYAKMPGYETVLTDFFWEMVHKDGVGNQQEEKGANHVVVK"
    site = "A70K"
    mutated_protein = wildtype_protein[:int(site[1:-1])-1] + site[-1] + wildtype_protein[int(site[1:-1]):]
    (vec, delta) = get_fused(model, wildtype_protein, mutated_protein, site)
    logger.info(f"Fused embeddings: {vec.shape}")
    logger.info(f"Mutation delta: {delta.shape}") 


def main():

    args = parse_args()
    logger = setup_logging(Path(args.log_dir), args.log_level)
    device = select_device(args.device)
    logger.info(f"Using device: {device}")
    model = create_model(args.config, device)
    check(model)

    test_fused_embeddings(model)
    
if __name__ == "__main__":
    #print("CWD:", os.getcwd())
    main()
