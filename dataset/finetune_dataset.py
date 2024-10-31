from torch.utils.data import Dataset
import pandas as pd
import json
import os

class MutaDescribeDataset(Dataset):
    def __init__(self, path, split="train", data_percent=1.0, **kwargs) -> None:
        """ path: path to dataset.csv """
        super().__init__()
        self.df = pd.read_csv(path)
        self.df = self.df[:int(len(self.df)*data_percent)]    # 5%
    
    def __getitem__(self, index):
        site = self.df["entry"][index].split("-")[1]
        prot1 = self.df["protein1"][index]
        prot2 = self.df["protein2"][index]
        uni_despt = self.df["uniprot_description"][index] if not pd.isna(self.df["uniprot_description"][index]) else ''
        GPT_despt = self.df["GPT_description"][index] if not pd.isna(self.df["GPT_description"][index]) else ''
        prot_function = self.df["function"][index]
        template = "Next is a feature of the mutation {} to {} at position {}. Please generate a {} text to describe it."
        mut_prompt = template.format(site[0], site[-1], int(site[1:-1]), "long detailed" if len(GPT_despt) >= 1 else "brief summary")
        return prot1, prot2, site, (uni_despt + ' ' + GPT_despt).strip(), prot_function, mut_prompt
    
    def __len__(self):
        return len(self.df)

    def get_example(self):
        for i in range(len(self)):
            prot1, prot2, site, desc, _, _ = self[i]
            yield "Wild Type:" + prot1[:20] + "...\t" + "Site: " + site + "\tMutation Effect: " + desc[:50] + "..."
        raise RuntimeError("Number of examples exceed dataset length!")