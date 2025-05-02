import argparse
import os
import torch
from PIL import Image
from torchvision import transforms as T
from typing import List
from transformers import CLIPProcessor, CLIPModel
import json


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


def collect_clip_metrics(real_dir, gen_dir, prompts_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip, proc = load_clip(device)

    real_emb, gen_emb, txt_emb = [], [], []

    for file_name in sorted(os.listdir(real_dir)):
        file_path = os.path.join(real_dir, file_name)
        if os.path.isfile(file_path):
            real_emb.append(get_clip_img_embedding(
                clip, proc, file_path, device))

    gen_files = sorted(
        f for f in os.listdir(gen_dir)
        if os.path.isfile(os.path.join(gen_dir, f))
    )
    for file_name in gen_files:
        file_path = os.path.join(gen_dir, file_name)
        gen_emb.append(get_clip_img_embedding(clip, proc, file_path, device))

    with open(prompts_json) as f:
        prm_json = json.load(f)

    if isinstance(prm_json, list):
        prompts = prm_json
        prompts = [prm_json[file_name] for file_name in gen_files]

    if len(prompts) != len(gen_emb):
        raise ValueError("prompts_json length must match #generated images")

    for t in prompts:
        txt_emb.append(get_clip_text_embedding(clip, proc, t, device))

    clip_i = avg_pairwise_cos(real_emb, gen_emb)
    clip_t = avg_pairwise_cos(gen_emb, txt_emb)

    print(f"CLIP-I (subject fidelity) : {clip_i:.4f}")
    print(f"CLIP-T (prompt  fidelity) : {clip_t:.4f}")

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
