import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import EsmTokenizer, EsmForMaskedLM, BertTokenizer, BertForMaskedLM

valid_aa = ['A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L',
        'O', 'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

class RandomModel(nn.Module):
    def __init__(self, path, device) -> None:
        super().__init__()
        self.protein_tokenizer = EsmTokenizer.from_pretrained(path)
        self.device = device

    def lm_design(self, protein, text, **kwargs):
        all_logits = []
        for p in protein:
            random_logits = torch.randn(len(p), 33)
            random_logits[:, :4] = -100
            random_logits[:, 24:] = -100
            all_logits.append(F.softmax(torch.cat(
                [torch.ones(1, 33) * -100, random_logits, torch.ones(1025 - len(p), 33) * -100],
                dim=0
            ), dim=0))
        return torch.stack(all_logits, dim=0).to(self.device)


class VanllinaEsm(nn.Module):
    def __init__(self, path, protein_maxlen, device, lambd=0.1, ontoprotein=False):
        super().__init__()
        self.protein_maxlen = protein_maxlen
        if ontoprotein:
            self.esm = BertForMaskedLM.from_pretrained(path)
            self.protein_tokenizer = BertTokenizer.from_pretrained(path)
        else:
            self.esm = EsmForMaskedLM.from_pretrained(path)
            self.protein_tokenizer = EsmTokenizer.from_pretrained(path)

        self.ontoprotein = ontoprotein
        self.device = device
        self.lambd = lambd
        self.loss_names = ["loss_reward", "loss_kl"]

    def forward(self, protein_mut, protein_wild, text, scores):
        outputs = self.predict_fitness(protein_mut, protein_wild, text, return_dict=True)
        pred_scores = outputs["score"]
        loss_reward = torch.tensor(0.).to(self.device)
        for i in range(len(protein_mut)):
            for j in range(i):
                if scores[i] > scores[j]:
                    loss_reward += -F.logsigmoid(pred_scores[i] - pred_scores[j])
                else:
                    loss_reward += -F.logsigmoid(pred_scores[j] - pred_scores[i])

        loss_fn = nn.KLDivLoss(reduction='none')
        logits = outputs["logits"]
        logits_orig = self.frozen_lm_head(self.protein_model.esm(outputs["mask_input_ids"], outputs["attention_mask"])[0])
        targets = F.softmax(logits_orig, dim=-1)
        i_indices = torch.arange(logits.shape[0]).unsqueeze(1).unsqueeze(2).expand(-1, logits.shape[1], logits.shape[2])
        j_indices = torch.arange(logits.shape[1]).unsqueeze(0).unsqueeze(2).expand(logits.shape[0], -1, logits.shape[2])
        k_indices = outputs["orig_input_ids"].unsqueeze(2).expand(-1, -1, logits.shape[2])
        loss_kl = torch.mean(loss_fn(logits[i_indices, j_indices, k_indices], targets[i_indices, j_indices, k_indices]) * outputs["attn_mask"])

        return loss_reward + self.lambd * loss_kl, {"loss_reward": loss_reward.detach(), "loss_kl": loss_kl.detach()}

    def validate_fn(self, protein_mut, protein_wild, text, scores):
        preds = self.predict_fitness(protein_mut, protein_wild)
        return preds, scores

    @torch.no_grad()
    def lm_design(self, protein, text, **kwargs):
        if self.ontoprotein:
            protein = [" ".join(list(p)) for p in protein]
        protein = self.protein_tokenizer(
            protein,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.protein_maxlen,
            return_tensors='pt'
        ).to(self.device)
        logits = self.esm(protein.input_ids, protein.attention_mask, return_dict=True).logits

        i_indices = torch.arange(logits.shape[0]).unsqueeze(1).unsqueeze(2).expand(-1, logits.shape[1], logits.shape[2])
        j_indices = torch.arange(logits.shape[1]).unsqueeze(0).unsqueeze(2).expand(logits.shape[0], -1, logits.shape[2])
        k_indices = protein.input_ids.unsqueeze(2).expand(-1, -1, logits.shape[2])
        logits -= logits[i_indices, j_indices, k_indices]
        logits[torch.where(protein.input_ids == self.protein_tokenizer.cls_token_id)] = -1000
        if not self.ontoprotein:
            logits[torch.where(protein.input_ids == self.protein_tokenizer.eos_token_id)] = -1000
        else:
            logits[torch.where(protein.input_ids == self.protein_tokenizer.sep_token_id)] = -1000
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                logits[i, j, protein.input_ids[i][j]] = -1000
        logits[(1 - protein.attention_mask).bool()] = -1000
        if not self.ontoprotein:
            logits[:, :, :4] = -1000
            logits[:, :, 24:] = -1000
        else:
            logits[:, :, :5] = -1000
            logits[:, :, 25:] = -1000
        return F.softmax(logits, dim=-1)


    def predict_fitness(self, protein, wild_type, *kwargs):
        mut_i_index, mut_j_index, mut_k_wt_index, mut_k_mt_index = [], [], [], []
        for i in range(len(protein)):
            assert(len(protein[i]) == len(wild_type[0]))
            for j in range(len(protein[i])):
                if protein[i][j] != wild_type[0][j]:
                    mut_i_index.append(i)
                    mut_j_index.append(j + 1)
                    mut_k_wt_index.append(wild_type[0][j])
                    mut_k_mt_index.append(protein[i][j])
        inp_protein = self.protein_tokenizer(
            protein,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.protein_maxlen,
            return_tensors='pt'
        ).to(self.device)
        mut_i_index = torch.LongTensor(mut_i_index).to(self.device)
        mut_j_index = torch.LongTensor(mut_j_index).to(self.device)
        mut_k_wt_index = self.protein_tokenizer.encode(mut_k_wt_index, add_special_tokens=False, return_tensors='pt').squeeze().to(self.device)
        mut_k_mt_index = self.protein_tokenizer.encode(mut_k_mt_index, add_special_tokens=False, return_tensors='pt').squeeze().to(self.device)
        # print(mut_i_index, mut_j_index, mut_k_wt_index, mut_k_mt_index)

        inp_protein.input_ids[mut_i_index, mut_j_index] = self.protein_tokenizer.mask_token_id

        logits = self.esm(**inp_protein, return_dict=True).logits
        logits = logits[mut_i_index, mut_j_index, mut_k_mt_index] - logits[mut_i_index, mut_j_index, mut_k_wt_index]
        mask = (mut_i_index.unsqueeze(1) == torch.arange(len(protein)).unsqueeze(0).to(self.device)).transpose(0, 1).float()
        score = logits * mask
        return score.sum(dim=-1)
    