# -*- coding: utf-8 -*-
"""
===============================================================================
GAN FINAL Y FUNCIONAL PARA GENERACIÓN DE AUTOMÓVILES
===============================================================================

DESCRIPCIÓN:
    Versión definitiva y funcional de una DCGAN (Deep Convolutional GAN) para
    generar imágenes de automóviles. Este script contiene arquitecturas de red
    corregidas y robustas que aseguran la compatibilidad de dimensiones
    durante el entrenamiento.

OBJETIVO:
    - Entrenar una GAN funcional sobre el dataset CIFAR-10 (categoría automóvil).
    - Generar 30 imágenes nuevas y de alta calidad de automóviles.
    - Demostrar una implementación correcta y estable de una GAN en PyTorch.

AUTOR: David Timana
CURSO: Visión por Computador - GANs
FECHA: 2024
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. CONFIGURACIÓN DEL PROYECTO
# =============================================================================
def set_seed(seed=42):
    """Fija las semillas para reproducibilidad."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Hiperparámetros clave
CONFIG = {
    "batch_size": 32,
    "latent_dim": 100,
    "lr": 0.0002,
    "beta1": 0.5,
    "epochs": 20, # Suficientes para ver resultados iniciales
    "num_final_images": 30,
    "image_size": 32,
    "channels": 3
}

# Crear directorios para los resultados
os.makedirs("results/final_generated", exist_ok=True)


# =============================================================================
# 2. PREPARACIÓN DEL DATASET
# =============================================================================
class CarDataset(Dataset):
    """Dataset personalizado que filtra solo los automóviles de CIFAR-10."""
    def __init__(self, transform=None):
        cifar_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        # La clase 'automobile' tiene la etiqueta 1 en CIFAR-10
        self.car_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 1]
        self.cifar_dataset = cifar_dataset
        print(f"🚗 Encontrados {len(self.car_indices)} automóviles en el dataset CIFAR-10.")

    def __len__(self):
        return len(self.car_indices)

    def __getitem__(self, idx):
        img, _ = self.cifar_dataset[self.car_indices[idx]]
        return img


# =============================================================================
# 3. ARQUITECTURA DE LA RED (MODELOS)
# =============================================================================

def weights_init(m):
    """Inicializa los pesos de las capas convolucionales y de normalización."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    Generador: Toma un vector latente (ruido) y lo transforma en una imagen.
    Usa capas ConvTranspose2d para 'deconvolucionar' el ruido a una imagen de 32x32.
    """
    def __init__(self, latent_dim, channels):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Entrada: Vector latente (Z)
            # Salida: (256, 4, 4)
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Capa 2: (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Capa 3: (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Capa 4: (64, 16, 16) -> (channels, 32, 32)
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh() # Normaliza la salida a [-1, 1]
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """
    Discriminador: Toma una imagen y la clasifica como real o falsa.
    Usa capas convolucionales para reducir la imagen a una única probabilidad.
    """
    def __init__(self, channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Entrada: (channels, 32, 32)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 2: (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 3: (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 4: (256, 4, 4) -> (1, 1, 1) - Probabilidad final
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# =============================================================================
# 4. FUNCIONES AUXILIARES (VISUALIZACIÓN Y GUARDADO)
# =============================================================================
def save_progress_images(generator, fixed_noise, epoch, device):
    """Guarda una grilla de imágenes para visualizar el progreso."""
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"Imágenes Generadas - Época {epoch}")
    plt.axis("off")
    plt.savefig(f"results/final_generated/progress_epoch_{epoch:03d}.png")
    plt.close()
    generator.train()

def generate_final_images(generator, latent_dim, num_images, device):
    """Genera y guarda el conjunto final de 30 imágenes."""
    print(f"\n🎨 Generando las {num_images} imágenes finales...")
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        final_images = generator(noise).detach().cpu()

    # Guardar cada imagen individualmente
    for i in range(num_images):
        img = final_images[i]
        torchvision.utils.save_image(img, f"results/final_generated/car_{i+1:02d}.png", normalize=True)

    # Guardar la grilla final
    grid = torchvision.utils.make_grid(final_images, nrow=6, padding=2, normalize=True)
    plt.figure(figsize=(12, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"{num_images} Automóviles Generados por la GAN (Final)")
    plt.axis("off")
    plt.savefig("results/final_generated/final_30_cars_grid.png")
    plt.close()
    print(f"✅ {num_images} imágenes guardadas en 'results/final_generated/'.")


# =============================================================================
# 5. BUCLE DE ENTRENAMIENTO
# =============================================================================
def train_gan():
    """Función principal que orquesta todo el proceso."""
    print("🚀 Iniciando el entrenamiento de la GAN Definitiva...")
    print("=" * 60)

    # --- Setup ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo de entrenamiento: {device}")

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = CarDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)

    # --- Modelos ---
    netG = Generator(CONFIG["latent_dim"], CONFIG["channels"]).to(device)
    netD = Discriminator(CONFIG["channels"]).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)
    print("✅ Modelos Generador y Discriminador inicializados.")

    # --- Pérdida y Optimizadores ---
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=CONFIG["lr"], betas=(CONFIG["beta1"], 0.999))

    # Ruido fijo para visualización consistente
    fixed_noise = torch.randn(64, CONFIG["latent_dim"], 1, 1, device=device)

    # Etiquetas para la función de pérdida
    real_label = 1.
    fake_label = 0.

    # --- Bucle Principal ---
    print(f"🎯 Comenzando entrenamiento por {CONFIG['epochs']} épocas...")
    G_losses = []
    D_losses = []

    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}/{CONFIG['epochs']}")
        for i, data in enumerate(progress_bar):
            # ---------------------------
            # (1) Actualizar red D: maximizar log(D(x)) + log(1 - D(G(z)))
            # ---------------------------
            ## Entrenar con imágenes reales
            netD.zero_grad()
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            ## Entrenar con imágenes falsas
            noise = torch.randn(b_size, CONFIG["latent_dim"], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # ---------------------------
            # (2) Actualizar red G: maximizar log(D(G(z)))
            # ---------------------------
            netG.zero_grad()
            label.fill_(real_label) # Las etiquetas falsas son reales para el costo del generador
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Guardar pérdidas y actualizar barra de progreso
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            progress_bar.set_postfix({
                'Loss_D': f'{errD.item():.4f}',
                'Loss_G': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f}/{D_G_z2:.4f}'
            })

        # Guardar imágenes de progreso al final de cada época
        save_progress_images(netG, fixed_noise, epoch + 1, device)

    print("🏁 Entrenamiento completado.")
    print("=" * 60)

    # --- Generación Final ---
    generate_final_images(netG, CONFIG["latent_dim"], CONFIG["num_final_images"], device)


if __name__ == '__main__':
    train_gan()
