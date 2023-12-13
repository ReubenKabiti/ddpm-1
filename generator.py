from models.unet import UNet
import torch
import torchvision
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def generate(schedular, num_images=1, model_path=None, w=28, h=28, device="cpu"):
    img = torch.randn(num_images, 1, w, h, device=device)
    model = UNet().to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    for T in tqdm(range(schedular.num_steps)):
        t = schedular.num_steps - T - 1
        if t == 0:
            break

        alpha = schedular.alpha[t]
        alpha_cumprod = schedular.alpha_cumprod[t]
        beta = schedular.alpha_cumprod[t]

        p_noise = model(img, t)
        mean_inside = img[-1:] - (1 - alpha)/((1 - alpha_cumprod).sqrt())*p_noise
        mean = 1/alpha.sqrt()*mean_inside
        std_div = beta.sqrt()
        img = mean + std_div*torch.randn_like(p_noise)
        
    return img.cpu()
            

def to_pil(image):
    """
    convert a torch image to a PIL image
    Args
        image:
            A BxCxHxW tensor with values in the range [-1, 1]
    """

    for i, img in enumerate(image):
        image[i] = (img - img.min())/(img.max() - img.min())
    
    image = torchvision.utils.make_grid(image)
    return ToPILImage()(image)