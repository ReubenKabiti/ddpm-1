import torch

class LinearSchedular:

    def __init__(self, num_steps, start_beta, end_beta, device="cpu"):
        self.num_steps = num_steps
        self.start_beta = start_beta
        self.end_beta = end_beta

        self.beta = torch.linspace(start_beta, end_beta, num_steps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_cumprod = self.alpha.cumprod(dim=0)
    
    def add_noise(self, image, t):
        device = image.device
        ac = self.alpha_cumprod[t]
        noise = torch.randn_like(image, device=device)
        noised_image = ac.sqrt()*image + (1 - ac).sqrt()*noise
        return noised_image, noise