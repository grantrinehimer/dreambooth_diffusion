# A Reimplementation of _DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation_

<!-- ## Environment

Try to use the environment.yaml to make a conda environment. Honestly, this might not work so if not just make sure you're using python 3.10 and install packages with pip as needed. You could also try the requirements.txt as well (although diffusers has to be installed manually through cloning the repo to your computer). -->

## Introduction
The purpose of this Git repository is to document our final project, which re-implements the DreamBooth method presented in the chosen paper[1]. Recent text-to-image models have shown incredible performance in generating high quality images given a text prompt. However, these models, given a reference set of subjects, lack the ability to generate accurate and consistent renditions in different contexts. Dreambooth personalizes pre-trained T2I models by fine-tuning them on just 3–5 images of a subject, using a rare token identifier and class-specific prompts to synthesize subjects. The paper [1] contributes a class-specific Prior Preservation Loss (PPL) that allows T2I models to generate a diverse and contextually appropriate image of a subject while effectively preserving its identity.

## Chosen Result
We extend Figure 12 and Table 3 from the original paper [1] to qualitatively compare our reimplementation [Figure 1] and quantitatively assess the effect of PPL on subject fidelity and output diversity—key factors in evaluating the success of subject-driven generation. 
## GitHub Contents
`code\dreambooth_finetune` - Contains the Huggingface script for dreambooth finetuning, adapted for use in a non-CLI setting and broken down into a class structure
`code\batch_dreambooth.py` - Main script for training models on subjects, config files are used to pass in training parameters and target directories
`code\batch_config.yaml` - Contains parameters for training on an entire batch of subjects
`code\training_config.yaml` - Contains parameters for training on a specific subject, includes all specific training parameters
`data` - Contains images of subjects as well as randomly generated class images for use in PPL
## Re-implementation Details

## Reproduction Steps

## Results/Insights

## Conclusion

## References
1. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., & Aberman, K. (2023). *DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation*. arXiv:2208.12242. [https://arxiv.org/abs/2208.12242](https://arxiv.org/abs/2208.12242)
2. P. von Platen et al., Diffusers: State-of-the-art diffusion models. GitHub, 2022. [Online]. Available: https://github.com/huggingface/diffusers

## Acknowledgements
