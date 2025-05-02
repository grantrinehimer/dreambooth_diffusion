"""Batch DreamBooth fine‑tune + inference helper (Namespace edition).

This script iterates over `data/<subject>/` folders, fine‑tunes SD‑1.5 for each
subject using your DreamBooth `trainer.main()`, then generates a handful of demo
images.

**Key change requested by the user:** we now build an *`argparse.Namespace`*
programmatically instead of converting a dict to CLI strings. This keeps the
call tree type‑safe and avoids a second parse.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from typing import Dict, Sequence

import gc
import torch
from diffusers import StableDiffusionPipeline

# Local DreamBooth helpers (uploaded by the user)
import dreambooth_finetune.arguments as arguments  # provides `parse_args` with full DreamBooth defaults
import dreambooth_finetune.trainer as trainer    # entry‑point for training

from dotenv import load_dotenv
import os
import pandas as pd
import yaml

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
import datetime
from datetime import datetime

# Also check for mps

def batch_train(batch_config_path: Path, training_config_path: Path):
    
    with batch_config_path.open("r") as file:
        batch_config = yaml.safe_load(file)

    with training_config_path.open("r") as file:
        training_config = yaml.safe_load(file)
    

    print(batch_config)
    csv_dir = batch_config['csv_dir']
    prompts_df = pd.read_csv(os.path.join(csv_dir, "prompts.csv"))
    subjects_df = pd.read_csv(os.path.join(csv_dir, "subjects.csv"))

    data_dir = Path(batch_config['data_dir'])

    for index, subject_folder in enumerate(sorted(data_dir.iterdir())):

        if index < 3:
            continue

        if not subject_folder.is_dir():
            continue

        name = subject_folder.name
        image_class = subjects_df.loc[subjects_df["subject_name"] == name, "class"].values[0]
        is_live = subjects_df.loc[subjects_df["subject_name"] == name, "live"].values[0]

        print(f"\n▶ Processing subject: {name}")

        model_out = Path(os.path.join(batch_config['models_output_dir'], name))
        model_out.mkdir(parents=True, exist_ok=True)

        instance_prompt = batch_config['training_prompt'].format(batch_config["rare_token"], image_class)
        print(instance_prompt)

        # 1) Prepare Namespace for DreamBooth
        config_dict = dict(
            instance_data_dir=str(subject_folder),
            instance_prompt=instance_prompt,
            output_dir=str(model_out),
            gradient_checkpointing=True,
            **training_config,
        )

        if training_config['with_prior_preservation']:
            ppl_base_dir = batch_config['ppl_dir']
            class_data_dir = os.path.join(ppl_base_dir, image_class)
            class_prompt = f"a {image_class}"
            config_dict = dict(
                class_data_dir = class_data_dir,
                class_prompt = class_prompt,
                **config_dict
            )
        
        input_args = to_cli_args(config_dict)
        args = arguments.parse_args(input_args=input_args)

        trainer.train(args)
        del args
        torch.cuda.empty_cache()
        gc.collect()

        # Get prompts based off if live
        prompts = list(prompts_df.loc[prompts_df["live"] == is_live, "prompt"])
        # 2) Generate demo images
        produce_demo_images(
            model_dir=model_out,
            prompts=prompts,
            subject_class=image_class,
            rare_token=batch_config["rare_token"],
            images_out_root=Path(batch_config['images_output_dir']),
            num_images_per_prompt=4
        )



@torch.inference_mode()
def produce_demo_images(
    model_dir: Path,
    prompts: List,
    subject_class: str,
    rare_token: str,
    images_out_root: Path,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    num_images_per_prompt: int = 4,
    device: str | torch.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Load a freshly-fine-tuned SD-1.5 DreamBooth model and generate images for each
    prompt.

    Parameters
    ----------
    model_dir : Path
        Directory that contains the fine-tuned model weights (same as `output_dir`
        you passed to the trainer).
    prompts : List
        A column of text prompts.  Every row will be rendered.
    images_out_root : Path
        Root directory where results are stored.  Images for this model are saved
        under  `<images_out_root>/<model_dir.name>/`.
    num_inference_steps : int, default 30
        Forward steps in DDIM / Euler sampler.
    guidance_scale : float, default 7.5
        Classifier-free guidance scale.
    num_images_per_prompt : int, default 4
        How many variants to sample for **each** prompt.
    device : str | torch.device
        Where to move the pipeline.
    """
    images_out_root = Path(images_out_root)
    images_out_root.mkdir(parents=True, exist_ok=True)
    subject_out_dir = images_out_root / model_dir.name
    subject_out_dir.mkdir(exist_ok=True)

    # 1. Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_dir).to(device)
    finished_prompts = [prompt.format(rare_token, subject_class) for prompt in prompts]

    



    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") 

    for idx, prompt in enumerate(finished_prompts):
        images = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt
            ).images
        for img_id in range(num_images_per_prompt):
            fname = (
                subject_out_dir
                / f"{idx:03d}_{img_id}_{timestamp}.png"
            )
            image = images[img_id]
            image.save(fname)
            print(f"   💾 Saved: {fname.relative_to(images_out_root)}")

    
    del pipe
    torch.cuda.empty_cache()
    gc.collect()





# Convert dict to CLI-style args: ["--key", "value", "--flag", ...]
def to_cli_args(cfg):
    args = []
    for k, v in cfg.items():
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)  # For action="store_true"
            # skip False (don't include the flag at all)
        elif v is not None:
            args.extend([f"--{k}", str(v)])
    return args



def main(argv: Sequence[str] | None = None):
    load_dotenv()
    batch_train(Path("batch_config.yaml"), Path("training_config.yaml"))


if __name__ == "__main__":
    main()
