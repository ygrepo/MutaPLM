import json
import os
import random
from torch.utils.data import Dataset
import re

class LiteratureDataset(Dataset):
    def __init__(self, path, **kwargs) -> None:
        super().__init__()
        filter_accs = [x.strip().split(" ")[1] for x in open("./data/filter_acc.txt", "r").readlines()]
        self.uniprot2pubmed = json.load(open(os.path.join(path, "uniprot_pubmed.json"), "r"))
        self.uniprot2seq = {}
        self.uniprot2func = {}
        uniprot_data = json.load(open(os.path.join(path, "uniprot_accession.json"), "r"))
        keys = set()
        for id in uniprot_data:
            for key in uniprot_data[id]:
                if key not in keys:
                    keys.add(key)
            if "Sequence" in uniprot_data[id] and id in self.uniprot2pubmed and id not in filter_accs:
                self.uniprot2seq[id] = uniprot_data[id]["Sequence"]
                if "Description" in uniprot_data[id]:
                    pattern1 = r'\(PubMed:\d+(, PubMed:\d+)*\)'
                    pattern2 = r'\(By similarity\)'
                    self.uniprot2func[id] = "; ".join([re.sub(pattern2, '', re.sub(pattern1, '', text)) for text in uniprot_data[id]["Description"]])
        self.uniprot_ids = list(self.uniprot2seq.keys())
        self.pubmed_corpus = {}
        with open(os.path.join(path, "corpus.jsonl"), "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                if data["title"] is not None and data["abstract"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["title"] + " " + data["abstract"]
                elif data["title"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["title"]
                elif data["abstract"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["abstract"]
        for id in self.uniprot2pubmed:
            for i, pubmed_id in enumerate(self.uniprot2pubmed[id]):
                if pubmed_id not in self.pubmed_corpus:
                    self.uniprot2pubmed[id].pop(i)

    def get_by_uniport(self, id):
        print(id)
        if id in self.uniprot2func:
            print("Function:", self.uniprot2func[id])
            print("---------------------------------------------")
        for j in self.uniprot2pubmed[id]:
            print(self.pubmed_corpus[j])

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, index):
        id = self.uniprot_ids[index]
        seq = self.uniprot2seq[id]
        text_id = random.sample(self.uniprot2pubmed[id], k=1)[0]
        return seq, self.pubmed_corpus[text_id]

    def get_example(self):
        for i in range(len(self)):
            seq, text = self[i]
            yield "Accession: " + self.uniprot_ids[i] + "\tSequence:" + seq[:30] + "...\tText:" + text[:100]
        raise RuntimeError("Number of examples exceed dataset length!")

if __name__ == "__main__":
    dataset = LiteratureDataset("./data/pubs")
    cnt = 0
    print(len(dataset))
    for i in range(len(dataset)):
        cnt += len(dataset.uniprot2pubmed[dataset.uniprot_ids[i]])
    print(cnt)
    print(dataset.get_by_uniport("B3VI55"))