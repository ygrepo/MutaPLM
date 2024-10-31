import json
import os
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

name2target = {
    "AAV": "Strongly increased viability for packaging of a DNA payload for gene therapy.",
    "AMIE": "Increase in activity.",
    "avGFP": "Leads to enhanced fluorescence at 37 degrees Celsius.",
    "E4B": "Enhances cleavage by caspase-6 and granzyme B.",
    "LGK": "Increase in activity.",
    "UBE2I": "Increased growth rescue rate at high temperature in a yeast strain."
}

name2prompt = {
    "AAV": "Capsid protein self-assembles to form an icosahedral capsid with a T=1 symmetry, about 22 nm in diameter, and consisting of 60 copies of three size variants of the capsid protein VP1, VP2 and VP3 which differ in their N-terminus. The capsid encapsulates the genomic ssDNA. Binds to host cell heparan sulfate and uses host ITGA5-ITGB1 as coreceptor on the cell surface to provide virion attachment to target cell. This attachment induces virion internalization predominantly through clathrin-dependent endocytosis. Binding to the host receptor also induces capsid rearrangements leading to surface exposure of VP1 N-terminus, specifically its phospholipase A2-like region and putative nuclear localization signal(s). VP1 N-terminus might serve as a lipolytic enzyme to breach the endosomal membrane during entry into host cell and might contribute to virus transport to the nucleus.",
    "AMIE": "Catalyzes the hydrolysis of short-chain aliphatic amides to their corresponding organic acids with release of ammonia.",
    "avGFP": "Energy-transfer acceptor. Its role is to transduce the blue chemiluminescence of the protein aequorin into green fluorescent light by energy transfer. Fluoresces in vivo upon receiving energy from the Ca(2+)-activated photoprotein aequorin.",
    "E4B": "Ubiquitin-protein ligase that probably functions as an E3 ligase in conjunction with specific E1 and E2 ligases. May also function as an E4 ligase mediating the assembly of polyubiquitin chains on substrates ubiquitinated by another E3 ubiquitin ligase. May regulate myosin assembly in striated muscles together with STUB1 and VCP/p97 by targeting myosin chaperone UNC45B for proteasomal degradation.",
    "LGK": "Levoglucosan kinase that catalyzes the transfer of a phosphate group from ATP to levoglucosan (1,6-anhydro-beta-D-glucopyranose, LG) to yield glucose 6-phosphate in the presence of magnesium ion and ATP. In addition to the canonical kinase phosphotransfer reaction, the conversion requires cleavage of the 1,6-anhydro ring to allow ATP-dependent phosphorylation of the sugar O-6 atom.",
    "UBE2I": "Accepts the ubiquitin-like proteins SUMO1, SUMO2, SUMO3, SUMO4 and SUMO1P1/SUMO5 from the UBLE1A-UBLE1B E1 complex and catalyzes their covalent attachment to other proteins with the help of an E3 ligase such as RANBP2, CBX4 and ZNF451. Can catalyze the formation of poly-SUMO chains. Necessary for sumoylation of FOXL2 and KAT5. Essential for nuclear architecture and chromosome segregation. Sumoylates p53/TP53 at 'Lys-386'. Mediates sumoylation of ERCC6 which is essential for its transcription-coupled nucleotide excision repair activity",
}

class FitnessDataset(Dataset):
    def __init__(self, path, split="valid", name=None, nshot=None, **kwargs):
        super().__init__()
        if os.path.exists(os.path.join(path, "wild_type.json")):
            self.wild_type = json.load(open(os.path.join(path, "wild_type.json"), "r"))["seq"]
        else:
            self.wild_type = json.load(open(os.path.join(path, "starting_sequence.json"), "r"))
        data = json.load(open(os.path.join(path, split + ".json"), "r"))
        self.data = []
        
        for i in range(len(data)):
            if len(data[i]["seq"]) == len(self.wild_type):
                self.data.append(data[i])
        if nshot is not None:
            perm = np.random.permutation(len(self.data))[:nshot]
            new_data = [self.data[i] for i in perm]
            self.data = new_data
        self.starting_sequence = json.load(open(os.path.join(path, "starting_sequence.json"), "r"))
        self.prompt = name2prompt[path.split("/")[-1]]
        self.target = name2target[path.split("/")[-1]]

    def __len__(self):
        return len(self.data)

    def get_example(self):
        for i in range(len(self)):
            yield "Sequence:" + self.data[i]["seq"][:30] + "...\tFitness:" + str(self.data[i]["fitness"][0]) + "\tNum mutations:" + str(self.data[i]["num_mutations"])
        raise RuntimeError("Number of examples exceed dataset length!")
    
    def __getitem__(self, index):
        return self.data[index]["seq"], self.data[index]["fitness"][0]