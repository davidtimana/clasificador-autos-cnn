#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generative Adversarial Network (GAN) para Generación de Imágenes de Automóviles
VERSIÓN OPTIMIZADA PARA GOOGLE COLAB

Este script implementa una Deep Convolutional Generative Adversarial Network (DCGAN)
para generar imágenes de automóviles utilizando el dataset CIFAR-10. Incluye:
- Carga y preparación del dataset CIFAR-10 (filtrado para automóviles)
- Implementación de arquitecturas Generator y Discriminator
- Entrenamiento adversarial con optimización de hiperparámetros
- Generación de imágenes sintéticas de alta calidad
- Visualización de resultados y progreso del entrenamiento

Autor: David Timana
Fecha: 2024
Curso: Visión por Computador - GANs
Versión: Google Colab Optimizada
"""

# %%
# =============================================================================
# PASO 1: IMPORTAR LAS BIBLIOTECAS NECESARIAS
# =============================================================================

# Verificar si estamos en Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("🚀 Detectado Google Colab - Configurando entorno...")
except ImportError:
    IN_COLAB = False
    print("💻 Ejecutando en entorno local")

# Importar bibliotecas principales de PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Importar bibliotecas para análisis de datos y manipulación
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

# Importar bibliotecas para manejo de advertencias
import warnings
warnings.filterwarnings('ignore')

# Configuraciones específicas para Colab
if IN_COLAB:
    # Montar Google Drive (opcional)
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("📁 Google Drive montado exitosamente")
    except:
        print("⚠️ No se pudo montar Google Drive - continuando sin él")
    
    # Configurar matplotlib para Colab
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

print("✓ Bibliotecas importadas exitosamente")

# %%
# =============================================================================
# PASO 2: CONFIGURACIÓN DEL PROYECTO Y HIPERPARÁMETROS
# =============================================================================

def configurar_proyecto():
    """
    Configura los hiperparámetros y parámetros del proyecto.
    Optimizado para Google Colab con detección automática de GPU.
    
    Returns:
        dict: Diccionario con la configuración del proyecto
    """
    print("\n" + "="*60)
    print("PASO 2: CONFIGURACIÓN DEL PROYECTO Y HIPERPARÁMETROS")
    print("="*60)
    
    # Configuración principal del proyecto
    CONFIG = {
        "batch_size": 64,           # Tamaño del lote para entrenamiento
        "latent_dim": 100,          # Dimensión del espacio latente (vector de ruido)
        "lr": 0.0002,              # Tasa de aprendizaje (estándar para GANs)
        "beta1": 0.5,              # Parámetro beta1 para optimizador Adam
        "epochs": 10,              # Número de épocas de entrenamiento (reducido para métricas rápidas)
        "num_final_images": 30,    # Número de imágenes finales a generar
        "image_size": 64,          # Tamaño de imagen de salida (64x64 píxeles)
        "channels": 3              # Número de canales de color (RGB)
    }
    
    # Configuraciones específicas para Colab
    if IN_COLAB:
        CONFIG["batch_size"] = 128  # Lotes más grandes en Colab con GPU
        CONFIG["epochs"] = 30       # Menos épocas para demostración
    
    # Crear directorios para los resultados
    if IN_COLAB:
        os.makedirs("/content/results/final_generated", exist_ok=True)
        CONFIG["results_path"] = "/content/results/final_generated"
    else:
        os.makedirs("results/final_generated", exist_ok=True)
        CONFIG["results_path"] = "results/final_generated"
    
    # Mostrar configuración
    print("=== CONFIGURACIÓN DEL PROYECTO ===")
    for key, value in CONFIG.items():
        print(f"{key}: {value}")
    
    print(f"\n✓ Directorios creados: {CONFIG['results_path']}")
    
    return CONFIG

# %%
# =============================================================================
# PASO 3: FUNCIÓN PARA REPRODUCIBILIDAD
# =============================================================================

def establecer_reproducibilidad(seed=42):
    """
    Establece las semillas para garantizar reproducibilidad de resultados.
    
    Args:
        seed (int): Semilla para la generación de números aleatorios
    """
    print("\n" + "="*60)
    print("PASO 3: ESTABLECER REPRODUCIBILIDAD")
    print("="*60)
    
    # Establecer semillas para PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Establecer semillas para NumPy y Random
    np.random.seed(seed)
    random.seed(seed)
    
    # Configuraciones adicionales de PyTorch para reproducibilidad
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Reproducibilidad establecida con semilla: {seed}")

# %%
# =============================================================================
# PASO 4: PREPARACIÓN DEL DATASET
# =============================================================================

class CarDataset(Dataset):
    """
    Dataset personalizado que filtra solo los automóviles del dataset CIFAR-10.
    
    Esta clase crea un dataset específico para automóviles, que corresponde
    a la clase 1 en el dataset CIFAR-10 original.
    """
    
    def __init__(self, transform=None):
        """
        Inicializa el dataset de automóviles.
        
        Args:
            transform: Transformaciones a aplicar a las imágenes
        """
        # Cargar el dataset CIFAR-10 completo
        cifar_dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        # Filtrar solo los automóviles (etiqueta 1 en CIFAR-10)
        self.car_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == 1]
        self.cifar_dataset = cifar_dataset
        
        print(f"🚗 Encontrados {len(self.car_indices)} automóviles en el dataset CIFAR-10.")

    def __len__(self):
        """Retorna el número total de automóviles en el dataset."""
        return len(self.car_indices)

    def __getitem__(self, idx):
        """
        Obtiene una imagen de automóvil por su índice.
        
        Args:
            idx (int): Índice de la imagen
            
        Returns:
            torch.Tensor: Imagen de automóvil
        """
        img, _ = self.cifar_dataset[self.car_indices[idx]]
        return img

def preparar_dataset(CONFIG):
    """
    Prepara el dataset de automóviles para el entrenamiento.
    
    Args:
        CONFIG (dict): Configuración del proyecto
        
    Returns:
        tuple: (dataloader, dataset)
    """
    print("\n" + "="*60)
    print("PASO 4: PREPARACIÓN DEL DATASET")
    print("="*60)
    
    # Definir las transformaciones para las imágenes
    transform = transforms.Compose([
        transforms.Resize(CONFIG["image_size"]),                    # Redimensionar a 64x64
        transforms.ToTensor(),                                      # Convertir a tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    # Normalizar a [-1, 1]
    ])
    
    # Crear el dataset de automóviles
    dataset = CarDataset(transform=transform)
    
    # Crear el DataLoader con configuración optimizada para Colab
    num_workers = 2 if IN_COLAB else 0
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("=== INFORMACIÓN DEL DATASET ===")
    print(f"Tamaño del dataset: {len(dataset)} imágenes")
    print(f"Tamaño del lote: {CONFIG['batch_size']}")
    print(f"Número de lotes por época: {len(dataloader)}")
    print(f"Resolución de imagen: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"Canales de color: {CONFIG['channels']} (RGB)")
    print(f"Workers: {num_workers}")
    print(f"Pin Memory: {torch.cuda.is_available()}")
    
    return dataloader, dataset

# %%
# =============================================================================
# PASO 5: FUNCIÓN DE INICIALIZACIÓN DE PESOS
# =============================================================================

def inicializar_pesos(m):
    """
    Inicializa los pesos de las capas convolucionales y de normalización.
    
    Esta función aplica una inicialización específica para GANs que ayuda
    a estabilizar el entrenamiento adversarial.
    
    Args:
        m: Módulo de PyTorch (capa de la red)
    """
    classname = m.__class__.__name__
    
    # Inicialización para capas convolucionales
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # Inicialización para capas de normalización por lotes
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# %%
# =============================================================================
# PASO 6: ARQUITECTURA DEL GENERADOR
# =============================================================================

class Generator(nn.Module):
    """
    Generador: Transforma un vector de ruido latente en una imagen realista.
    
    La arquitectura utiliza capas de convolución transpuesta (ConvTranspose2d)
    para realizar un proceso de "upsampling" progresivo, transformando un
    vector de ruido de 100 dimensiones en una imagen de 64x64 píxeles.
    """
    
    def __init__(self, latent_dim, channels):
        """
        Inicializa la arquitectura del generador.
        
        Args:
            latent_dim (int): Dimensión del espacio latente
            channels (int): Número de canales de salida (RGB)
        """
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # Capa 1: Vector latente -> (512, 4, 4)
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Capa 2: (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Capa 3: (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Capa 4: (128, 16, 16) -> (64, 32, 32)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Capa 5: (64, 32, 32) -> (channels, 64, 64)
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Normaliza la salida a [-1, 1]
        )

    def forward(self, input):
        """
        Forward pass del generador.
        
        Args:
            input (torch.Tensor): Vector de ruido latente
            
        Returns:
            torch.Tensor: Imagen generada
        """
        return self.main(input)

# %%
# =============================================================================
# PASO 7: ARQUITECTURA DEL DISCRIMINADOR
# =============================================================================

class Discriminator(nn.Module):
    """
    Discriminador: Clasifica imágenes como reales o generadas.
    
    La arquitectura utiliza capas convolucionales para reducir progresivamente
    la resolución de la imagen hasta obtener una única probabilidad que
    indica si la imagen es real (1) o generada (0).
    """
    
    def __init__(self, channels):
        """
        Inicializa la arquitectura del discriminador.
        
        Args:
            channels (int): Número de canales de entrada (RGB)
        """
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # Capa 1: (channels, 64, 64) -> (64, 32, 32)
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 2: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 3: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 4: (256, 8, 8) -> (512, 4, 4)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Capa 5: (512, 4, 4) -> (1, 1, 1) - Probabilidad final
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # Produce una probabilidad entre 0 y 1
        )

    def forward(self, input):
        """
        Forward pass del discriminador.
        
        Args:
            input (torch.Tensor): Imagen de entrada
            
        Returns:
            torch.Tensor: Probabilidad de que la imagen sea real
        """
        return self.main(input)

# %%
# =============================================================================
# PASO 8: FUNCIONES AUXILIARES PARA VISUALIZACIÓN Y MÉTRICAS
# =============================================================================

def calcular_metricas_gan(generator, discriminator, dataloader, device, num_samples=1000):
    """
    Calcula métricas de calidad para evaluar el rendimiento de la GAN.
    
    Args:
        generator: Modelo generador
        discriminator: Modelo discriminador
        dataloader: DataLoader con imágenes reales
        device: Dispositivo de cómputo
        num_samples: Número de muestras para calcular métricas
        
    Returns:
        dict: Diccionario con las métricas calculadas
    """
    print("\n📊 Calculando métricas de calidad de la GAN...")
    
    generator.eval()
    discriminator.eval()
    
    with torch.no_grad():
        # Métricas del discriminador
        real_confidences = []
        fake_confidences = []
        
        # Recolectar confianzas del discriminador
        for i, real_data in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break
                
            real_data = real_data.to(device)
            batch_size = real_data.size(0)
            
            # Confianza en imágenes reales
            real_conf = discriminator(real_data).cpu().numpy()
            real_confidences.extend(real_conf.flatten())
            
            # Confianza en imágenes generadas
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = generator(noise)
            fake_conf = discriminator(fake_data).cpu().numpy()
            fake_confidences.extend(fake_conf.flatten())
        
        # Calcular métricas
        real_confidences = np.array(real_confidences[:num_samples])
        fake_confidences = np.array(fake_confidences[:num_samples])
        
        # Métricas de confianza
        avg_real_conf = np.mean(real_confidences)
        avg_fake_conf = np.mean(fake_confidences)
        std_real_conf = np.std(real_confidences)
        std_fake_conf = np.std(fake_confidences)
        
        # Métrica de separabilidad (cuán bien distingue el discriminador)
        separability = avg_real_conf - avg_fake_conf
        
        # Métrica de estabilidad (varianza de las confianzas)
        stability = (std_real_conf + std_fake_conf) / 2
        
        # Métrica de balance (cuán equilibradas están las confianzas)
        balance = 1 - abs(avg_real_conf - (1 - avg_fake_conf))
        
        # Métrica de calidad general
        quality_score = (separability * 0.4 + (1 - stability) * 0.3 + balance * 0.3)
        
        metricas = {
            'avg_real_confidence': avg_real_conf,
            'avg_fake_confidence': avg_fake_conf,
            'std_real_confidence': std_real_conf,
            'std_fake_confidence': std_fake_conf,
            'separability': separability,
            'stability': stability,
            'balance': balance,
            'quality_score': quality_score
        }
        
        print("=== MÉTRICAS DE CALIDAD ===")
        print(f"Confianza promedio en imágenes reales: {avg_real_conf:.4f}")
        print(f"Confianza promedio en imágenes generadas: {avg_fake_conf:.4f}")
        print(f"Separabilidad (real - fake): {separability:.4f}")
        print(f"Estabilidad (menor = mejor): {stability:.4f}")
        print(f"Balance: {balance:.4f}")
        print(f"Puntuación de calidad general: {quality_score:.4f}")
        
        return metricas

def guardar_metricas(metricas, epoch, results_path):
    """
    Guarda las métricas en un archivo CSV para seguimiento.
    
    Args:
        metricas (dict): Diccionario con las métricas
        epoch (int): Número de época
        results_path (str): Ruta para guardar las métricas
    """
    import csv
    import os
    
    csv_path = os.path.join(results_path, "metricas_gan.csv")
    
    # Crear archivo si no existe
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + list(metricas.keys()))
    
    # Agregar métricas de la época actual
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch] + list(metricas.values()))

def guardar_imagenes_progreso(generator, fixed_noise, epoch, device, results_path):
    """
    Guarda una grilla de imágenes para visualizar el progreso del entrenamiento.
    
    Args:
        generator: Modelo generador
        fixed_noise: Ruido fijo para visualización consistente
        epoch (int): Número de época actual
        device: Dispositivo de cómputo (CPU/GPU)
        results_path: Ruta para guardar las imágenes
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    # Crear grilla de imágenes
    grid = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)
    
    # Visualizar y guardar
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"Imágenes Generadas - Época {epoch}")
    plt.axis("off")
    
    # Guardar imagen
    save_path = os.path.join(results_path, f"progress_epoch_{epoch:03d}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Mostrar en Colab si es necesario
    if IN_COLAB and epoch % 10 == 0:  # Mostrar cada 10 épocas
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.title(f"Progreso - Época {epoch}")
        plt.axis("off")
        plt.show()
    
    generator.train()

def generar_imagenes_finales(generator, latent_dim, num_images, device, results_path):
    """
    Genera y guarda el conjunto final de imágenes sintéticas.
    
    Args:
        generator: Modelo generador entrenado
        latent_dim (int): Dimensión del espacio latente
        num_images (int): Número de imágenes a generar
        device: Dispositivo de cómputo (CPU/GPU)
        results_path: Ruta para guardar las imágenes
    """
    print(f"\n🎨 Generando las {num_images} imágenes finales...")
    
    generator.eval()
    with torch.no_grad():
        # Generar ruido aleatorio
        noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
        
        # Generar imágenes
        final_images = generator(noise).detach().cpu()

    # Guardar cada imagen individualmente
    for i in range(num_images):
        img = final_images[i]
        save_path = os.path.join(results_path, f"car_{i+1:02d}.png")
        torchvision.utils.save_image(img, save_path, normalize=True)

    # Guardar la grilla final
    grid = torchvision.utils.make_grid(
        final_images, 
        nrow=6, 
        padding=2, 
        normalize=True
    )
    
    # Mostrar y guardar la grilla final
    plt.figure(figsize=(15, 12))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.title(f"{num_images} Automóviles Generados por la GAN (Final)")
    plt.axis("off")
    
    # Guardar grilla
    grid_path = os.path.join(results_path, "final_30_cars_grid.png")
    plt.savefig(grid_path, bbox_inches='tight', dpi=150)
    
    # Mostrar en Colab
    if IN_COLAB:
        plt.show()
    else:
        plt.close()
    
    print(f"✅ {num_images} imágenes guardadas en '{results_path}'.")

def visualizar_metricas(results_path):
    """
    Visualiza las métricas guardadas durante el entrenamiento.
    
    Args:
        results_path (str): Ruta donde están guardadas las métricas
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    csv_path = os.path.join(results_path, "metricas_gan.csv")
    
    if not os.path.exists(csv_path):
        print("⚠️ No se encontraron métricas para visualizar.")
        return
    
    # Cargar métricas
    df = pd.read_csv(csv_path)
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Métricas de Calidad de la GAN durante el Entrenamiento', fontsize=16)
    
    # Gráfico 1: Confianzas del discriminador
    axes[0, 0].plot(df['epoch'], df['avg_real_confidence'], 'b-', label='Imágenes Reales', linewidth=2)
    axes[0, 0].plot(df['epoch'], df['avg_fake_confidence'], 'r-', label='Imágenes Generadas', linewidth=2)
    axes[0, 0].set_title('Confianza del Discriminador')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Confianza Promedio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gráfico 2: Separabilidad
    axes[0, 1].plot(df['epoch'], df['separability'], 'g-', linewidth=2)
    axes[0, 1].set_title('Separabilidad (Real - Fake)')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Separabilidad')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gráfico 3: Estabilidad
    axes[1, 0].plot(df['epoch'], df['stability'], 'orange', linewidth=2)
    axes[1, 0].set_title('Estabilidad (Menor = Mejor)')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Estabilidad')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gráfico 4: Puntuación de Calidad General
    axes[1, 1].plot(df['epoch'], df['quality_score'], 'purple', linewidth=2)
    axes[1, 1].set_title('Puntuación de Calidad General')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Puntuación')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    save_path = os.path.join(results_path, "metricas_evolucion.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if IN_COLAB:
        plt.show()
    else:
        plt.close()
    
    print(f"📊 Gráfico de métricas guardado en: {save_path}")
    
    # Mostrar resumen de métricas finales
    ultima_fila = df.iloc[-1]
    print("\n=== RESUMEN DE MÉTRICAS FINALES ===")
    print(f"Confianza en imágenes reales: {ultima_fila['avg_real_confidence']:.4f}")
    print(f"Confianza en imágenes generadas: {ultima_fila['avg_fake_confidence']:.4f}")
    print(f"Separabilidad: {ultima_fila['separability']:.4f}")
    print(f"Estabilidad: {ultima_fila['stability']:.4f}")
    print(f"Balance: {ultima_fila['balance']:.4f}")
    print(f"Puntuación de calidad general: {ultima_fila['quality_score']:.4f}")

# %%
# =============================================================================
# PASO 9: BUCLE PRINCIPAL DE ENTRENAMIENTO
# =============================================================================

def entrenar_gan(CONFIG):
    """
    Función principal que orquesta todo el proceso de entrenamiento de la GAN.
    
    Args:
        CONFIG (dict): Configuración del proyecto
    """
    print("\n" + "="*60)
    print("PASO 9: BUCLE PRINCIPAL DE ENTRENAMIENTO")
    print("="*60)
    
    print("🚀 Iniciando el entrenamiento de la GAN Definitiva...")
    print("=" * 60)

    # --- Configuración inicial ---
    establecer_reproducibilidad(42)
    
    # Detectar dispositivo (optimizado para Colab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Dispositivo de entrenamiento: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Preparación del dataset ---
    dataloader, dataset = preparar_dataset(CONFIG)

    # --- Inicialización de modelos ---
    print("\n=== INICIALIZACIÓN DE MODELOS ===")
    netG = Generator(CONFIG["latent_dim"], CONFIG["channels"]).to(device)
    netD = Discriminator(CONFIG["channels"]).to(device)
    
    # Aplicar inicialización de pesos
    netG.apply(inicializar_pesos)
    netD.apply(inicializar_pesos)
    print("✅ Modelos Generador y Discriminador inicializados.")

    # --- Configuración de optimizadores y función de pérdida ---
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizerD = optim.Adam(
        netD.parameters(), 
        lr=CONFIG["lr"], 
        betas=(CONFIG["beta1"], 0.999)
    )
    optimizerG = optim.Adam(
        netG.parameters(), 
        lr=CONFIG["lr"], 
        betas=(CONFIG["beta1"], 0.999)
    )

    # --- Ruido fijo para visualización consistente ---
    fixed_noise = torch.randn(64, CONFIG["latent_dim"], 1, 1, device=device)

    # --- Etiquetas para la función de pérdida ---
    real_label = 1.
    fake_label = 0.

    # --- Bucle principal de entrenamiento ---
    print(f"\n🎯 Comenzando entrenamiento por {CONFIG['epochs']} épocas...")
    G_losses = []
    D_losses = []

    for epoch in range(CONFIG["epochs"]):
        progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}/{CONFIG['epochs']}")
        
        for i, data in enumerate(progress_bar):
            # ---------------------------
            # (1) ACTUALIZAR RED DISCRIMINADOR
            # Maximizar log(D(x)) + log(1 - D(G(z)))
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
            # (2) ACTUALIZAR RED GENERADOR
            # Maximizar log(D(G(z)))
            # ---------------------------
            netG.zero_grad()
            label.fill_(real_label)  # Las etiquetas falsas son reales para el costo del generador
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # --- Guardar pérdidas y actualizar barra de progreso ---
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            progress_bar.set_postfix({
                'Loss_D': f'{errD.item():.4f}',
                'Loss_G': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f}/{D_G_z2:.4f}'
            })

        # --- Calcular y guardar métricas cada 5 épocas ---
        if (epoch + 1) % 5 == 0:
            metricas = calcular_metricas_gan(netG, netD, dataloader, device, num_samples=500)
            guardar_metricas(metricas, epoch + 1, CONFIG["results_path"])
        
        # --- Guardar imágenes de progreso al final de cada época ---
        guardar_imagenes_progreso(netG, fixed_noise, epoch + 1, device, CONFIG["results_path"])

    print("🏁 Entrenamiento completado.")
    print("=" * 60)

    # --- Calcular métricas finales ---
    print("\n📊 Calculando métricas finales...")
    metricas_finales = calcular_metricas_gan(netG, netD, dataloader, device, num_samples=1000)
    guardar_metricas(metricas_finales, CONFIG["epochs"], CONFIG["results_path"])
    
    # --- Generación final de imágenes ---
    generar_imagenes_finales(netG, CONFIG["latent_dim"], CONFIG["num_final_images"], device, CONFIG["results_path"])
    
    # --- Visualizar métricas ---
    visualizar_metricas(CONFIG["results_path"])

# %%
# =============================================================================
# PASO 10: FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de la GAN.
    """
    print("🎯 GENERATIVE ADVERSARIAL NETWORK (GAN) PARA AUTOMÓVILES")
    print("=" * 60)
    print("Implementación completa de DCGAN con PyTorch")
    print("Dataset: CIFAR-10 (filtrado para automóviles)")
    print("Objetivo: Generar 30 imágenes sintéticas de automóviles")
    print("Versión: Google Colab Optimizada")
    print("=" * 60)
    
    # Configurar el proyecto
    CONFIG = configurar_proyecto()
    
    # Ejecutar el entrenamiento
    entrenar_gan(CONFIG)
    
    print("\n🎉 ¡Proceso completado exitosamente!")
    print(f"📁 Revisa la carpeta '{CONFIG['results_path']}' para ver las imágenes generadas.")
    
    if IN_COLAB:
        print("💡 Para descargar las imágenes, usa el panel de archivos de Colab")

# %%
# =============================================================================
# EJECUCIÓN DEL SCRIPT
# =============================================================================

if __name__ == '__main__':
    main()
