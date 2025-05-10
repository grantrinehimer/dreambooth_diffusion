# A Reimplementation of _DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation_

The original paper can be referenced [here](https://arxiv.org/pdf/2208.12242).

[Dataset](https://github.com/google/dreambooth)


## Environment

Try to use the environment.yaml to make a conda environment. Honestly, this might not work so if not just make sure you're using python 3.10 and install packages with pip as needed. You could also try the requirements.txt as well (although diffusers has to be installed manually through cloning the repo to your computer).

## Introduction

## Chosen Result
`code\dreambooth_finetune` - Contains the Huggingface script for dreambooth finetuning, adapted for use in a non-CLI setting and broken down into a class structure
`code\batch_dreambooth.py` - Main script for training models on subjects, config files are used to pass in training parameters and target directories
`code\batch_config.yaml` - Contains parameters for training on an entire batch of subjects
`code\training_config.yaml` - Contains parameters for training on a specific subject, includes all specific training parameters
`data` - Contains images of subjects as well as randomly generated class images for use in PPL
## GitHub
