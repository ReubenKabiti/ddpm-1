from trainer import Trainer
from models.unet import UNet
from schedulars.linear import LinearSchedular
from generator import generate, to_pil
import torch


class Pipeline:

    """
    A pipeline is an object that sets up the training loop if required, loads an already saved model, sets up the
    device to use, and can generate an image if requested
    """


class SimplePipeline(Pipeline):

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = UNet().to(self.device)
        self.schedular = LinearSchedular(400, 0.0001, 0.002, device=self.device)

    def train(self, loader, checkpoint_dir, epochs=100):
        trainer = Trainer(self.unet, self.schedular, checkpoint_dir=checkpoint_dir, device=self.device)
        trainer.train(loader, epochs)

    def load_from_file(self, path):
        self.checkpoint_dir = path

    def generate(self, num_images=1):
        imgs = generate(self.schedular, num_images, model_path=self.checkpoint_dir, device=self.device)
        imgs = to_pil(imgs)
        return imgs