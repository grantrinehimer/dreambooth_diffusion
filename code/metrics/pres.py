import torch
import argparse
import os
from PIL import Image
from torchvision import transforms as T

"""
Prior Preservation Metric (PRES)
"""


def load_dino(device):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    model.eval().to(device)
    return model


def get_embedding(model, image_dir, device):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])  # specific to the DINO pretrained model I'm using
    image = Image.open(image_dir).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, 'forward_features'):
            feats = model.forward_features(x)
        else:
            out = model(x)
            feats = out[0] if isinstance(out, (tuple, list)) else out
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))
        elif feats.ndim == 3:
            feats = feats[:, 0, :]
        feats = feats / feats.norm(dim=1, keepdim=True)
    return feats.cpu()


def calculate_pres(model, real_dir, gen_dir, device):
    embeddings = {"real": [], "gen": []}
    for fname in sorted(os.listdir(real_dir)):
        path = os.path.join(real_dir, fname)
        if os.path.isfile(path):
            embeddings["real"].append(get_embedding(model, path, device))
    for fname in sorted(os.listdir(gen_dir)):
        path = os.path.join(gen_dir, fname)
        if os.path.isfile(path):
            embeddings["gen"].append(get_embedding(model, path, device))

    embeddings = {"real": torch.cat(
        embeddings["real"], dim=0), "gen": torch.cat(embeddings["gen"], dim=0)}

    pairwise_cosine_similarities = embeddings["real"] @ embeddings["gen"].t()
    return pairwise_cosine_similarities.mean().item()


def collect_pres(real_dir, gen_dir, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_dino(device)
    pres = calculate_pres(model, real_dir, gen_dir, device)
    return pres


def main():
    """
    parser = argparse.ArgumentParser(description='Compute PRES metric')
    parser.add_argument('--real_dir', type=str, required=True,
                        help='Path to directory of real subject images')
    parser.add_argument('--gen_dir',  type=str, required=True,
                        help='Path to directory of generated prior-class images')
    args = parser.parse_args()

    pres = collect_pres(args.real_dir, args.gen_dir)
    print(f'Prior Preservation Metric (PRES): {pres:.4f}')

    """
