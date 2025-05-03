import argparse
import os
import torch
from PIL import Image
from torchvision import transforms as T
from typing import List
from transformers import CLIPProcessor, CLIPModel
import pandas as pd


def load_clip(device: torch.device):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32").to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, proc


def get_clip_img_embedding(model, processor, img_path: str, device):
    img = Image.open(img_path).convert("RGB")

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(feat, p=2, dim=1).cpu()


def get_clip_text_embedding(model, processor, text: str, device):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        feat = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(feat, p=2, dim=1).cpu()


def avg_pairwise_cos(A: List[torch.Tensor], B: List[torch.Tensor]) -> float:
    A = torch.cat(A, 0)   # (N,D)
    B = torch.cat(B, 0)   # (M,D)
    return (A @ B.T).mean().item()


def collect_clip_metrics(real_dir, gen_dir, prompt_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, proc = load_clip(device)

    real_emb, gen_emb, txt_emb = [], [], []

    for file_name in sorted(os.listdir(real_dir)):
        file_path_real = os.path.join(real_dir, file_name)
        if os.path.isfile(file_path_real):
            real_emb.append(get_clip_img_embedding(
                clip, proc, file_path_real, device))

    for file_name in sorted(os.listdir(gen_dir)):
        file_path_gen = os.path.join(gen_dir, file_name)
        if os.path.isfile(file_path_gen):
            gen_emb.append(get_clip_img_embedding(
                clip, proc, file_path_gen, device))

    prompts = pd.read_csv(prompt_dir)

    for t in prompts:
        txt_emb.append(get_clip_text_embedding(clip, proc, t, device))

    clip_i = avg_pairwise_cos(real_emb, gen_emb)
    clip_t = avg_pairwise_cos(gen_emb, txt_emb)

    return clip_i, clip_t


def main():
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", required=True,
                        help="Directory with reference (real) subject images")
    parser.add_argument("--gen_dir",  required=True,
                        help="Directory with generated images")
    parser.add_argument("--prompts_json", required=True,
                        help="JSON list or dict of prompts, len == #generated images")
    args = parser.parse_args()

    clip_i, clip_t = collect_clip_metrics(
        args.real_dir, args.gen_dir, args.prompts_json)
    """
