# A Reimplementation of _DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation_

<!-- ## Environment

Try to use the environment.yaml to make a conda environment. Honestly, this might not work so if not just make sure you're using python 3.10 and install packages with pip as needed. You could also try the requirements.txt as well (although diffusers has to be installed manually through cloning the repo to your computer). -->

## Introduction
The purpose of this Git repository is to document our final project, which re-implements the DreamBooth method presented in the chosen paper[1]. Recent text-to-image (T2I) models have shown incredible performance in generating high quality images given a text prompt. However, these models, given a reference set of subjects, lack the ability to generate accurate and consistent renditions in different contexts. Dreambooth personalizes pre-trained T2I models by fine-tuning them on just 3–5 images of a subject, using a rare token identifier and class-specific prompts to synthesize subjects. The paper [1] contributes a class-specific Prior Preservation Loss (PPL) that allows T2I models to generate a diverse and contextually appropriate image of a subject while effectively preserving its identity.

## Chosen Result
We extend Figure 12 and Table 3 from the original paper [1] to qualitatively compare our reimplementation [Figure 1] and quantitatively assess the effect of PPL on subject fidelity and output diversity—key factors in evaluating the success of subject-driven generation. 
## GitHub Contents
`code\dreambooth_finetune` - Contains the Huggingface script for dreambooth finetuning, adapted for use in a non-CLI setting and broken down into a class structure
`code\batch_dreambooth.py` - Main script for training models on subjects, config files are used to pass in training parameters and target directories
`code\batch_config.yaml` - Contains parameters for training on an entire batch of subjects
`code\training_config.yaml` - Contains parameters for training on a specific subject, includes all specific training parameters
`data` - Contains images of subjects as well as randomly generated class images for use in PPL
## Re-implementation Details
This repository contains code and evaluation scripts to reproduce the quantitative results of our DreamBooth reimplementation (Stable Diffusion v1.5) and compare them against the original DreamBooth (Imagen) metrics. The evaluation produces per-class and per-condition (no‐PPL vs. PPL) scores for PRES, DIV, DINO, CLIP-I, and CLIP-T, and outputs summary CSV tables and an averaged comparison.

## 📁 Repository Structure

```
├── data/                     # Real reference images and metadata
│   ├── subjects.csv          # subject_name,class,live flag
|   └── ppl/                  # TODO GRANT EXPLAIN THIS
│       └── <subject_name>/…
|   └── subjects/             # real image folders (e.g. data/subjects/dog/00.jpg,...)
│       └── <subject_name>/…
│
├── results/                  # Generated images by condition
│   ├── no_ppl/               # baseline generations (no prior preservation)
│   │   └── <subject_name>/…  
│   └── ppl/                  # generations with Prior Preservation Loss
│       └── <subject_name>/…
│
├── code/                     # Evaluation scripts
|   ├── metrics/              # Metric scripts
│   │   └── pres.py           # computes PRES metric (and DINO metric)!
│   │   └── div.py            # computes DIV metric
│   │   └──clip_embeddings.py # computes CLIP‑I and CLIP‑T
│   └── evaluation.ipynb      # notebook to aggregate & export metrics


│ TODO: ADD INFO ON DREAMBOOTH_FINETUNE FOLDER 

├── requirements.txt          # Python dependencies
└── README.md                 # this file
```

## Reproduction Steps

1. **Clone the repo**:

   ```bash
   git clone https://github.com/grantrinehimer/dreambooth_diffusion.git
   cd <repo>
   ```
2. **Create a conda environment** (Python 3.10 recommended):

   ```bash
   conda create -n myenv python=3.10
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Setup Accelerate Config**

   ```bash
   accelerate config
   ```
   We recommend enabling fp16 mixed precision.

5. **Use stable_diffusion_test.ipynb to download your base pretrained model.**

   We use stable diffusion v1.5. Ensure that the model is located at the directory referred to in training_config.yaml.
   
6. **Run training script**
   ```bash
   accelerate launch batch_dreambooth.py
   ```
   This script will generate a model for every subject, so beware of disk space. It also runs inference on the fine-tuned models to generate the result images. In training_config.yaml, you can configure training parameters. We used gradient checkpointing, mixed precision FP16, and 8-bit Adam. You will also see an option to enable PPL and the number of images to generate for PPL. We used 100 images and created models both with and without PPL. Remember to change the model and images output directories in batch_config.yaml when generating PPL so as to not overwrite the models without PPL. We ran the script once with and without PPL for 400 training steps. We also used a batch size of 1.

6. **Execute all cells in code/evaluation.ipynb**:

   * The notebook imports `pres.py`, `div.py`, and `clip_embeddings.py`.
   * It iterates over each subject & condition, computes all five metrics, and writes per-metric CSVs:
     * `pres_results.csv`
     * `div_results.csv`
     * `clip_results.csv`
     * `dino_results.csv`
   * The final cell **aggregates** these into a single summary table, saving `all_results.csv` that contains mean PRES, DIV, DINO, CLIP-I, CLIP-T for `non-ppl` and `ppl`.
   * 
## Results/Insights

Across all metrics, our reimplementation reproduces the original paper’s trends—with slightly lower absolute values owing to our reduced prompt set (8 vs. 25) and shorter fine-tuning (400 vs. 1000 steps). This is highlighted below [Table 1]; the same trends persist between the Dreambooth metrics [1] and ours. 

In both cases, adding PPL sharply reduces prior collapse (lower PRES), meaning the model no longer “hallucinates” the fine-tuned subject when generating random class samples. We interpreted this as a lack of overfitting to the original subject; PPL aids in understanding the key components of what features make up the class without recreating the original subject, which is evident when prompts contain another subject from the same class.

Moreover, PPL boosts sample diversity under both pipelines, as generated images vary more in pose, background, and articulation. Nevertheless, our significant difference in DIV scores in PPL and no-PPL of 0.245 and 0.207 respectively, are indicative of our reduced amount of output images per prompt compared to the Dreambooth [1] output (2 vs. 4): having more output images mitigates average variance. When you only have two images, that one distance completely determines the mean; with four images, you average over six distances, which smooths out any outliers and reduces the overall DIV value.

Importantly, even with only 2 outputs per prompt and 400 training steps, PPL maintained its benefits: relative reductions in PRES and gains in DIV closely match those reported by Ruiz et al. [1]. This robustness suggests that class-specific prior preservation can be deployed under constrained compute budgets without losing its ability to preserve subject identity and encourage diverse generations.

![image title](image_name.png)

## Conclusion

## References
1. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. arXiv:2208.12242. [https://arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242)
2. P. von Platen et al., Diffusers: State-of-the-art diffusion models. GitHub, 2022. [Online]. Available: https://github.com/huggingface/diffusers
3. Mathilde Caron, Hugo Touvron, Ishan Misra, Herve ́ Je ́gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021.

## Acknowledgements
