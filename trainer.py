import torch
import numpy as np
from torch import nn, optim
class Trainer:

    def __init__(self, model, schedular, checkpoint_dir=None, device="cpu"):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.schedular = schedular
    
    def train(self, loader, epochs=10, lr=3e-4):
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            for batch, x in enumerate(loader):
                x = x.to(self.device)
                t = np.random.randint(0, self.schedular.num_steps)
                noisy, noise = self.schedular.add_noise(x, t)
                y = self.model(noisy)
                loss = loss_fn(y, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"epoch {epoch+1}/{epochs} batch {batch+1}/{len(loader)} --- loss: {loss.item()}")
                if self.checkpoint_dir:
                    torch.save(self.model.state_dict(), self.checkpoint_dir)


