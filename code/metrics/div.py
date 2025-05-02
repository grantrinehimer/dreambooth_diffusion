import os
import torch
from PIL import Image
import torchvision.transforms as T
import lpips

"""
Diversity Metric (DIV)
"""


def load_image_tensor(path, device):
    lpips_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    img = Image.open(path).convert('RGB')
    x = lpips_transform(img).unsqueeze(0).to(device)
    x = (x * 2.0) - 1.0
    return x


def compute_diversity(paths, device=None, net='alex'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images, got {len(paths)}")

    imgs = [load_image_tensor(p, device) for p in paths]

    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(len(imgs)):
            for j in range(i+1, len(imgs)):
                total += loss_fn(imgs[i], imgs[j]).item()
                count += 1

    return total / count


def collect_div(gen_dir, device=None, net='alex'):
    return compute_diversity(gen_dir, device=device, net=net)
