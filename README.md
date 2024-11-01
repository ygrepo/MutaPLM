## MutaPLM

This the the official repository for the NeurIPS 2024 paper [MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering](https://arxiv.org/abs/2410.22949).

#### Requirements

```bash
pytorch==1.13.1+cu117
transformers==4.36.1
peft==0.9.0
pandas
numpy
scipy
evoprotgrad
nltk
rouge_score
sequence_models
scikit-learn
```

#### Data

The pre-training dataset and the **MutaDescribe** dataset are available at [HuggingFace](https://huggingface.co/datasets/icycookies/MutaDescribe). Download the data and place them under the `data` folder.

#### Model Checkpoints

Before running the scripts, you should:
- Download the PLM checkpoint [esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and put it in `ckpts/esm2-650m`.
- Download the LLM checkpoint [BioMedGPT-LM](https://huggingface.co/PharMolix/BioMedGPT-LM-7B) and put it in `ckpts/biomedgpt-lm`. If you intend to perform evaluation only, you can just download the configuration files.
- Download the fine-tuned checkpoint [MutaPLM](https://huggingface.co/PharMolix/MutaPLM) and put it in `ckpts/mutaplm`. 


#### Implementation

For pre-training on protein literature, run the following script:

```bash
bash scripts/train/pretrain.sh
```

For fine-tuning on the MutaDescribe dataset, run the following script:

```bash
bash scripts/train/finetune.sh
```

For evaluating MutaPLM on mutation explanation, run the following script:

```bash
bash scripts/test/mutaplm_explain.sh
```

For evaluating MutaPLM on mutation engineering, run the following script:

```bash
bash scripts/test/mutaplm_engineer.sh
```

#### Citation
```
@misc{luo2024mutaplm,
      title={MutaPLM: Protein Language Modeling for Mutation Explanation and Engineering}, 
      author={Yizhen Luo and Zikun Nie and Massimo Hong and Suyuan Zhao and Hao Zhou and Zaiqing Nie},
      year={2024},
      eprint={2410.22949},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.22949}, 
}
```