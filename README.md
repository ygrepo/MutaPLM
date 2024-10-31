## MutaPLM

This the the implementation code for **MutaPLM: rotein Language Modeling for Mutation Explanation and Engineering.**

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

#### MutaDescribe

The constructed **MutaDescribe** is displayed in `data/mutadescribe` which involves several csv files. We provide the mutation entry (entry column), the wild-type protein (protein1 column), the mutant (protein2 column), the function of the wild-type (function column), and the description of mutational effects (all_description column). Due to size limits, we provide the test sets and 100 samples for the training and validation set. The whole dataset will be publicly released soon.


#### Checkpoints

Before running the scripts, you should download:
- protein model checkpoint: [esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and place it under `ckpts/esm2-650m`.
- language model checkpoint: [BioMedGPT-LM](https://huggingface.co/PharMolix/BioMedGPT-LM-7B) and place it under `ckpts/biomedgpt-lm`.
- our checkpoint of MutaPLM and place it under `ckpts/mutaplm`. The model weight will be publicly released soon.


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

For evaluating MutaPLM on mutation explanation, run the following script:

```bash
bash scripts/test/mutaplm_engineer.sh
```

For the fitness optimization experiments, run the following script:

```bash
bash scripts/optimize/mutaplm.sh
```

