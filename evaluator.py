from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

from collections import OrderedDict
import copy
import yaml
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import EsmForMaskedLM, EsmTokenizer, BertForMaskedLM, BertTokenizer

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from dataset import dataset_name2cls
from dataset.fitness_dataset import name2prompt, name2target
from model import model_name2cls
from model.esm_landscape import EsmForLandscapeRegression

class Evaluator(ABC):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--dataset_name", type=str, default="mutadescribe")
        parser.add_argument("--dataset_path", type=str, default="./data/")
        parser.add_argument("--model_name", type=str, default="mutaplm")
        parser.add_argument("--model_config_path", type=str, default="./configs/mutaplm_inference.yaml")
        parser.add_argument("--model_checkpoint", type=str, default=None)
        parser.add_argument("--pred_save_path", type=str, default="./outputs/pred.txt")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--device", type=int, default=0)
        return parser

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.device = torch.device("cuda", self.args.device)
        # self.device = torch.device("cpu")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self._setup_data()
        self._setup_model()

    def _setup_data(self):
        logger.info(f"Loading dataset {self.args.dataset_name}...")
        self.dataset = dataset_name2cls[self.args.dataset_name](self.args.dataset_path)
        logger.info(f"Num Samples: {len(self.dataset)}",)
        if hasattr(self.dataset, "get_example"):
            for i, example in enumerate(self.dataset.get_example()):
                if i >= 2:
                    break
                logger.info(example)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
        )

    def _setup_model(self):
        logger.info("Loading model...")
        model_cls = model_name2cls[self.args.model_name]
        model_cfg = yaml.load(open(self.args.model_config_path, "r"), Loader=yaml.Loader)
        model_cfg["device"] = self.device
        self.model = model_cls(**model_cfg).to(self.device)

        if self.args.model_checkpoint is not None:
            logger.info(f"Load model checkpoint from {self.args.model_checkpoint}")
            state_dict = torch.load(open(self.args.model_checkpoint, "rb"), map_location="cpu")
            new_ckpt = state_dict["model"]
            print(self.model.load_state_dict(new_ckpt, strict=False))

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError

class MutaExplainEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__(args)

    def evaluate(self):
        logger.info("Start evaluation!")
        self.model.eval()
        all_preds_func, all_labels_func = [], []
        all_preds_mut, all_labels_mut = [], []
        all_preds_func_tokens, all_labels_func_tokens = [], []
        all_preds_mut_tokens, all_labels_mut_tokens = [], []
        meteor_func, meteor_mut = [], []
        with open(self.args.pred_save_path, "w") as f:
            f.write("Site\tPred_Func\tLabel_Func\tPred_Effect\tLabel_Effect\n")
        for i, data in enumerate(tqdm(self.dataloader)):
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16):
                    # print(self.model.forward_ft(*data))
                    preds_func, preds_mut = self.model.generate(data[0], data[1], data[3], pfunction=data[4])
                    # if i <= 1:
                    #     print(data[-1])
                    #     print(preds_func, data[4])
                    #     print(preds_mut, data[3])
                    for j in range(len(data[-1])):
                        all_preds_func.append(preds_func[j])
                        all_labels_func.append(data[4][j])
                        all_preds_mut.append(preds_mut[j])
                        all_labels_mut.append(data[3][j])
                        with open(self.args.pred_save_path, "a+") as f:
                            f.write(f"{data[2][j]}\t{preds_func[j]}\t{data[4][j]}\t{preds_mut[j]}\t{data[3][j]}\n")
                        all_preds_func_tokens.append(preds_func[j].split(" "))
                        all_labels_func_tokens.append([data[4][j].split(" ")])
                        meteor_func.append(meteor_score(all_labels_func_tokens[-1], all_preds_func_tokens[-1]))
                        all_preds_mut_tokens.append(preds_mut[j].split(" "))
                        all_labels_mut_tokens.append([data[3][j].split(" ")])
                        meteor_mut.append(meteor_score(all_labels_mut_tokens[-1], all_preds_mut_tokens[-1]))
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        scores_func, scores_mut = [], []
        for i in range(len(all_preds_func)):
            scores_func.append(scorer.score(all_preds_func[i], all_labels_func[i]))
            scores_mut.append(scorer.score(all_preds_mut[i], all_labels_mut[i]))
        bleu2_func = corpus_bleu(all_labels_func_tokens, all_preds_func_tokens, weights=(0.5, 0.5))
        bleu4_func = corpus_bleu(all_labels_func_tokens, all_preds_func_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        print("------------Function--------------")
        print("BLEU-2 = %.4lf" % bleu2_func)
        print("BLEU-4 = %.4lf" % bleu4_func)
        print("METEOR = %.4lf" % np.mean(meteor_func))
        print("ROUGE-1 = %.4lf" % (np.mean([rs['rouge1'].fmeasure for rs in scores_func])))
        print("ROUGE-2 = %.4lf" % (np.mean([rs['rouge2'].fmeasure for rs in scores_func])))
        print("ROUGE-L = %.4lf" % (np.mean([rs['rougeL'].fmeasure for rs in scores_func])))
        print("------------Mutation--------------")
        bleu2_mut = corpus_bleu(all_labels_mut_tokens, all_preds_mut_tokens, weights=(0.5, 0.5))
        bleu4_mut = corpus_bleu(all_labels_mut_tokens, all_preds_mut_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        print("BLEU-2 = %.4lf" % bleu2_mut)
        print("BLEU-4 = %.4lf" % bleu4_mut)
        print("METEOR = %.4lf" % np.mean(meteor_mut))
        print("ROUGE-1 = %.4lf" % (np.mean([rs['rouge1'].fmeasure for rs in scores_mut])))
        print("ROUGE-2 = %.4lf" % (np.mean([rs['rouge2'].fmeasure for rs in scores_mut])))
        print("ROUGE-L = %.4lf" % (np.mean([rs['rougeL'].fmeasure for rs in scores_mut])))

class MutaEngineerEvaluator(Evaluator):
    def __init__(self, args):
        super().__init__(args)

    def evaluate(self):
        logger.info("Start evaluation!")
        self.model.eval()
        all_preds = []
        all_preds_with_pos = []
        all_labels = []
        for data in tqdm(self.dataloader):
            with torch.no_grad():
                with autocast(dtype=torch.bfloat16):
                    preds = self.model.lm_design(data[0], data[3], pfunction=data[-2], muta_prompt=data[-1])
                    all_labels += data[2]
                    all_preds.append(preds)
                    pos = torch.tensor([int(x[1:-1]) for x in data[2]])
                    preds = preds[torch.arange(len(data[-1])), pos]
                    all_preds_with_pos.append(torch.argmax(preds, dim=1))
        all_preds = torch.cat(all_preds, dim=0).flatten(1, 2)
        all_preds_with_pos = "".join(self.model.protein_tokenizer.decode(torch.cat(all_preds_with_pos)).split(" "))

        all_pos = []
        all_aa = []
        for i in range(len(all_preds)):
            top50 = all_preds[i].topk(50).indices
            all_pos.append(top50 // len(self.model.protein_tokenizer))
            all_aa.append(top50 % len(self.model.protein_tokenizer))
        all_aa = self.model.protein_tokenizer.batch_decode(torch.stack(all_aa, dim=0))
        
        acc, rec = 0, 0
        with open(self.args.pred_save_path, "w") as f:
            f.write("Labels\tPreds\tSequence\n")
            for i in range(len(all_preds)):
                seq = self.dataset[i][0]
                all_aa[i] = "".join(all_aa[i].split(" "))
                preds = []
                for j in range(50):
                    pos = all_pos[i][j].item()
                    preds.append(seq[pos - 1] + str(pos) + all_aa[i][j])
                f.write(all_labels[i] + "\t" + ",".join(preds[:10]) + "\t" + self.dataset[i][0] + "\n")
                if all_labels[i] in preds:
                    rec += 1
                if all_preds_with_pos[i] == all_labels[i][-1]:
                    acc += 1
        print("Accuracy = ", acc / len(all_labels))
        print("Recall@50 = ", rec / len(all_labels))

class FitnessOptimizeEvaluator(Evaluator):
    @staticmethod
    def add_arguments(parser):
        parser = Evaluator.add_arguments(parser)
        parser.add_argument("--surrogate_path", type=str, default="./ckpts/landscape_ckpts/")
        parser.add_argument("--num_candidates", type=int, default=100)
        parser.add_argument("--num_rounds", type=int, default=10)
        parser.add_argument("--score_save_path", type=str, default="./outputs/")
        parser.add_argument("--evo_prot_grad", action="store_true")
        return parser

    def __init__(self, args) -> None:
        super().__init__(args)
        self._setup_surrogate()
        self.prompt = name2prompt[self.args.dataset_name]
        self.target = name2target[self.args.dataset_name]
        if self.args.evo_prot_grad:
            import evo_prot_grad
            self.expert = evo_prot_grad.get_expert(
                expert_name="bert",
                model=BertForMaskedLM.from_pretrained("./ckpts/protein_ckpts/ontoprotein"),
                tokenizer=BertTokenizer.from_pretrained("./ckpts/protein_ckpts/ontoprotein"),
                device=self.device,
                temperature=1.0
            )

    def _setup_surrogate(self):
        self.surrogate = EsmForLandscapeRegression("./ckpts/protein_ckpts/esm1b", self.args.surrogate_path, self.device)
        self.surrogate.to(self.device)
        self.surrogate.eval()

    def evaluate(self):
        logger.info("Start evaluation!")
        self.model.eval()

        mx_scores, mean_scores = [], []
        cur_fitness = 0
        for i in tqdm(range(20)):
            protein = [self.dataset.starting_sequence]
            prev_scores = torch.tensor([0.0])
            cur_fitness = self.surrogate(protein).item()
            print("Initial protein fitness:", cur_fitness)
            print("Function:", self.prompt)
            print("Target:", self.target)
            if self.args.evo_prot_grad:
                import evo_prot_grad
                cur_mx_scores, cur_mean_scores = [], []
                all_proteins = [[] for j in range(self.args.num_rounds)]
                for j in range(self.args.num_candidates // 10):
                    new_proteins, scores = evo_prot_grad.DirectedEvolution(
                        n_steps=10,
                        max_mutations=self.args.num_rounds + 1,
                        wt_protein=protein[0],
                        parallel_chains=10,
                        experts=[self.expert],
                        output='all',
                        random_seed=i+42
                    )()
                    for round in range(self.args.num_rounds):
                        all_proteins[round] += ["".join(p.split(" ")) for p in new_proteins[round]]

                for round in range(self.args.num_rounds):
                    round_scores = []
                    with torch.no_grad():
                        for batch in range((self.args.num_candidates - 1) // self.args.batch_size + 1):
                            st, ed = batch * self.args.batch_size, min(len(all_proteins[round]), (batch + 1) * self.args.batch_size)
                            round_scores.append(self.surrogate(all_proteins[round][st:ed]))
                    round_scores = torch.cat(round_scores, dim=0)
                    cur_mx_scores.append(torch.max(round_scores).item())
                    if round >= 1:
                        cur_mx_scores[-1] = max(cur_mx_scores[-1], cur_mx_scores[-2])
                    cur_mean_scores.append(torch.mean(round_scores).item())
                mx_scores.append(cur_mx_scores)
                mean_scores.append(cur_mean_scores)
            else:
                torch.random.manual_seed(i)
                cur_mx_scores, cur_mean_scores = [], []
                for round in range(self.args.num_rounds):
                    with torch.no_grad():
                        with autocast(dtype=torch.bfloat16):
                            all_preds = []
                            for batch in range((len(protein) - 1) // self.args.batch_size + 1):
                                st, ed = batch * self.args.batch_size, min(len(protein), (batch + 1) * self.args.batch_size)
                                preds = self.model.lm_design(
                                    protein[st:ed], 
                                    muta_prompt=["Not Available"] * (ed - st),
                                    pfunction=[self.prompt] * (ed - st), 
                                    text=[self.target] * (ed - st), 
                                    use_gt_function=True
                                )
                                preds += prev_scores[st:ed].to(self.device).view(ed - st, 1, 1).expand(ed - st, preds.shape[1], preds.shape[2])
                                all_preds.append(preds)
                            preds = torch.cat(all_preds, dim=0)
                            topk = torch.multinomial(preds.flatten(), self.args.num_candidates)
                            # topk = torch.topk(preds.flatten(), self.args.num_candidates)
                            indices = topk
                            idx = indices // (preds.shape[1] * 33)
                            pos = indices % (preds.shape[1] * 33) // 33
                            aa = self.model.protein_tokenizer.batch_decode(indices % 33)
                            print(pos, aa)
                            prev_scores = preds.flatten()[indices]
                    new_protein = []
                    for j in range(self.args.num_candidates):
                        cur = protein[idx[j].item()]
                        new_protein.append(cur[:pos[j].item() - 1] + aa[j] + cur[pos[j].item():])

                    protein = new_protein
                    round_scores = []
                    with torch.no_grad():
                        for batch in range((self.args.num_candidates - 1) // self.args.batch_size + 1):
                            st, ed = batch * self.args.batch_size, min(len(protein), (batch + 1) * self.args.batch_size)
                            round_scores.append(self.surrogate(protein[st:ed]))
                    round_scores = torch.cat(round_scores, dim=0)
                    print(protein, round_scores)
                    cur_mx_scores.append(torch.max(round_scores).item())
                    if round >= 1:
                        cur_mx_scores[-1] = max(cur_mx_scores[-1], cur_mx_scores[-2])
                    cur_mean_scores.append(torch.mean(round_scores).item())
                mx_scores.append(cur_mx_scores)
                mean_scores.append(cur_mean_scores)
        mx_scores = np.array(mx_scores)
        print("Max scores:")
        for i in range(self.args.num_rounds):
            print("Round ", i, " Fitness=", np.mean(mx_scores[:, i]), "\pm", np.var(mx_scores[:, i]))
        mean_scores = np.array(mean_scores)
        print("Avg scores:")
        for i in range(self.args.num_rounds):
            print("Round ", i, " Fitness=", np.mean(mean_scores[:, i]), "\pm", np.var(mx_scores[:, i]))
        with open(self.args.score_save_path, "a") as f:
            f.write(self.args.dataset_name + "\t")
            f.write("%.4lf" % (cur_fitness) + "\t")
            for i in range(self.args.num_rounds):
                f.write("%.4lf" % (np.mean(mx_scores[:, i])) + "\t" + "%.4lf" % (np.var(mx_scores[:, i])) + "\t")
            f.write("\n")