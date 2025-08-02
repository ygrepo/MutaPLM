from model.vanllina_esm import VanllinaEsm, RandomModel
from model.mutaplm import MutaPLM

model_name2cls = {
    "esm": VanllinaEsm,
    "mutaplm": MutaPLM,
    "random": RandomModel
}