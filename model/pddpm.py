import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import random
from torch import Tensor
from noise import snoise2
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, in_channels)
        self.silu = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(1, in_channels)

    def forward(self, x):
        identity = x
        out = self.norm1(self.conv1(x))
        out = self.silu(out)
        out = self.norm2(self.conv2(out))
        out += identity
        return out

class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Unet, self).__init__()
        # Downsample
        self.down1 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(64)
        self.down2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.res2 = ResidualBlock(128)
        self.down3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.res3 = ResidualBlock(256)
        self.up1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.res4 = ResidualBlock(128)
        self.up2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.res5 = ResidualBlock(64)
        self.up3 = nn.ConvTranspose2d(64, out_channels, 4, stride=2, padding=1)

    def forward(self, x):
        x = F.silu(self.down1(x))
        x = self.res1(x)
        x = F.silu(self.down2(x))
        x = self.res2(x)
        x = F.silu(self.down3(x))
        x = self.res3(x)

        x = F.silu(self.up1(x))
        x = self.res4(x)
        x = F.silu(self.up2(x))
        x = self.res5(x)
        x = self.up3(x)
        return x


def add_structured_simplex_noise(image, intensity):
    scale = 1
    original_device = image.device
    image = image.detach().cpu()
    image_np = image.numpy()

    height, width = image_np.shape[-2:]
    noise = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise[y, x] = snoise2(x * scale, y * scale)

    noised_image_np = image_np + noise * intensity

    noised_image = torch.from_numpy(noised_image_np).float().to(original_device)

    return noised_image


def process_patch(model, patch, noise_level):
    noised_patch = add_structured_simplex_noise(patch, intensity=noise_level)
    denoised_patch = model.unet(noised_patch)
    return denoised_patch


class pDDPM(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.unet = Unet()
        self.patch_size = config['target_size'][0] // 2
        # self.noise_levels = [0.1, 0.2, 0.3, 0.4] 
        self.noise_levels = [0.3] 

    def forward(self, x):
        _, _, h, w = x.size()
        reconstructed = torch.zeros_like(x)
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                for noise_level in self.noise_levels:
                    patch = process_patch(self, patch, noise_level)
                reconstructed[:, :, i:i+self.patch_size, j:j+self.patch_size] = patch
        return reconstructed

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch
        _, _, h, w = x.size()
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
        noise_level = random.choice(self.noise_levels)
        noised_patch = add_structured_simplex_noise(patch, intensity=noise_level)
        denoised_patch = self.unet(noised_patch)
        loss = F.mse_loss(denoised_patch, y[:, :, i:i+self.patch_size, j:j+self.patch_size])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch 
        _, _, h, w = x.size()
        reconstructed = torch.zeros_like(x)
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = x[:, :, i:i+self.patch_size, j:j+self.patch_size]
                for noise_level in self.noise_levels:
                    patch = process_patch(self, patch, noise_level)
                reconstructed[:, :, i:i+self.patch_size, j:j+self.patch_size] = patch
        loss = F.mse_loss(reconstructed, y)
        self.log('val_loss', loss)
        return loss

    def detect_anomaly(self, x: Tensor):

        rec = self.forward(x)
        anomaly_map = torch.abs(x - rec)
        anomaly_score = torch.sum(anomaly_map, dim=(1, 2, 3))
        return {
            'reconstruction': rec,
            'anomaly_map': anomaly_map,
            'anomaly_score': anomaly_score
        }

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001)