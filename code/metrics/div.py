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


def compute_diversity(gen_dir, device=None, net='alex'):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    loss_fn = lpips.LPIPS(net=net).to(device).eval()

    img_files = sorted([
        os.path.join(gen_dir, f)
        for f in os.listdir(gen_dir)
        if os.path.isfile(os.path.join(gen_dir, f))
    ])
    imgs = [load_image_tensor(p, device) for p in img_files]

    n = len(imgs)
    if n < 2:
        raise ValueError(
            f"Need ≥2 images to compute diversity; found {n} in {gen_dir}")
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i+1, n):
            with torch.no_grad():
                d = loss_fn(imgs[i], imgs[j]).item()
            total += d
            count += 1

    div = total / count
    return div


def collect_div(gen_dir, device=None, net='alex'):
    return compute_diversity(gen_dir, device=device, net=net)
