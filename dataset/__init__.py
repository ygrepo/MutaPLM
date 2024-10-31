from dataset.literature_dataset import LiteratureDataset
from dataset.finetune_dataset import MutaDescribeDataset
from dataset.fitness_dataset import FitnessDataset

dataset_name2cls = {
    "literature": LiteratureDataset,
    "mutadescribe": MutaDescribeDataset,
    "AAV": FitnessDataset,
    "AMIE": FitnessDataset,
    "avGFP": FitnessDataset,
    "E4B": FitnessDataset,
    "LGK": FitnessDataset,
    "UBE2I": FitnessDataset,
}