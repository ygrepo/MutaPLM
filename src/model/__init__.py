from src.model.vanilla_esm import VanllinaEsm, RandomModel
from src.model.mutaplm import MutaPLM

model_name2cls = {
    "esm": VanllinaEsm,
    "mutaplm": MutaPLM,
    "random": RandomModel
}