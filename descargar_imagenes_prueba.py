# -*- coding: utf-8 -*-
"""
Script para descargar 100 imágenes de prueba
Para evaluar los modelos de clasificación de autos
Autor: David Timana | Curso: Visión por Computador
"""

import os
import requests
import urllib.request
from PIL import Image
import numpy as np
import time
import random

print("📥 DESCARGADOR DE IMÁGENES DE PRUEBA")
print("=" * 50)

# Crear directorio para las imágenes
IMAGES_DIR = "imagenes_prueba"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Crear subdirectorios por clase
classes = ['avion', 'automovil', 'pajaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion']
for class_name in classes:
    class_dir = os.path.join(IMAGES_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

print(f"✅ Directorios creados en: {IMAGES_DIR}")

# =============================================================================
# 1. DESCARGAR DESDE UNSPLASH (GRATUITO)
# =============================================================================
print("\n1. Descargando desde Unsplash...")

def download_from_unsplash():
    """Descargar imágenes desde Unsplash usando su API pública"""
    
    # URLs de Unsplash para diferentes clases
    unsplash_urls = {
        'automovil': [
            'https://images.unsplash.com/photo-1549317661-bd32c8ce0db2?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1580273916550-e323be2ae537?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1544636331-e26879cd4d9b?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1563720223185-11003d516935?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1549924231-f129b911e442?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400&h=400&fit=crop'
        ],
        'avion': [
            'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400&h=400&fit=crop'
        ],
        'pajaro': [
            'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1444464666168-49d633b86797?w=400&h=400&fit=crop'
        ],
        'gato': [
            'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop'
        ],
        'perro': [
            'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=400&h=400&fit=crop',
            'https://images.unsplash.com/photo-1517423440428-a5a00ad493e8?w=400&h=400&fit=crop'
        ]
    }
    
    downloaded_count = 0
    
    for class_name, urls in unsplash_urls.items():
        class_dir = os.path.join(IMAGES_DIR, class_name)
        
        for i, url in enumerate(urls):
            try:
                # Descargar imagen
                filename = f"{class_name}_{i+1:02d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                urllib.request.urlretrieve(url, filepath)
                
                # Verificar que la imagen se descargó correctamente
                with Image.open(filepath) as img:
                    img.verify()
                
                print(f"   ✅ {filename} descargada")
                downloaded_count += 1
                
                # Pausa para no sobrecargar el servidor
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Error descargando {class_name}_{i+1}: {e}")
    
    return downloaded_count

# =============================================================================
# 2. CREAR IMÁGENES SINTÉTICAS
# =============================================================================
print("\n2. Creando imágenes sintéticas...")

def create_synthetic_images():
    """Crear imágenes sintéticas para completar las 100 imágenes"""
    
    synthetic_count = 0
    
    for class_name in classes:
        class_dir = os.path.join(IMAGES_DIR, class_name)
        
        # Contar imágenes existentes
        existing_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # Crear imágenes sintéticas para llegar a 10 por clase
        needed = max(0, 10 - existing_count)
        
        for i in range(needed):
            try:
                # Crear imagen sintética simple
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                
                # Agregar algunos patrones según la clase
                if class_name == 'automovil':
                    # Patrones rectangulares para autos
                    img_array[100:150, 50:200] = [255, 0, 0]  # Rojo
                elif class_name == 'avion':
                    # Patrones triangulares para aviones
                    img_array[50:100, 100:150] = [0, 0, 255]  # Azul
                elif class_name == 'gato':
                    # Patrones circulares para gatos
                    center_y, center_x = 112, 112
                    y, x = np.ogrid[:224, :224]
                    mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
                    img_array[mask] = [255, 255, 0]  # Amarillo
                
                # Guardar imagen
                filename = f"{class_name}_synthetic_{i+1:02d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                img = Image.fromarray(img_array)
                img.save(filepath, 'JPEG')
                
                print(f"   ✅ {filename} creada")
                synthetic_count += 1
                
            except Exception as e:
                print(f"   ❌ Error creando {class_name}_synthetic_{i+1}: {e}")
    
    return synthetic_count

# =============================================================================
# 3. DESCARGAR DESDE PIXABAY (GRATUITO)
# =============================================================================
print("\n3. Descargando desde Pixabay...")

def download_from_pixabay():
    """Descargar imágenes desde Pixabay"""
    
    # URLs de Pixabay (imágenes gratuitas)
    pixabay_urls = {
        'automovil': [
            'https://cdn.pixabay.com/photo/2015/05/28/23/12/auto-788747_640.jpg',
            'https://cdn.pixabay.com/photo/2016/02/13/13/11/cuba-1197800_640.jpg',
            'https://cdn.pixabay.com/photo/2016/11/18/12/51/automobile-1834274_640.jpg',
            'https://cdn.pixabay.com/photo/2017/03/27/14/56/auto-2179220_640.jpg',
            'https://cdn.pixabay.com/photo/2017/07/31/11/21/people-2557396_640.jpg'
        ],
        'avion': [
            'https://cdn.pixabay.com/photo/2014/05/18/11/26/airplane-344942_640.jpg',
            'https://cdn.pixabay.com/photo/2016/11/18/17/20/airplane-1836415_640.jpg',
            'https://cdn.pixabay.com/photo/2017/08/06/12/06/people-2604149_640.jpg',
            'https://cdn.pixabay.com/photo/2018/01/15/07/51/woman-3083379_640.jpg',
            'https://cdn.pixabay.com/photo/2019/03/09/17/30/airplane-4045959_640.jpg'
        ]
    }
    
    downloaded_count = 0
    
    for class_name, urls in pixabay_urls.items():
        class_dir = os.path.join(IMAGES_DIR, class_name)
        
        for i, url in enumerate(urls):
            try:
                filename = f"{class_name}_pixabay_{i+1:02d}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                urllib.request.urlretrieve(url, filepath)
                
                # Verificar imagen
                with Image.open(filepath) as img:
                    img.verify()
                
                print(f"   ✅ {filename} descargada")
                downloaded_count += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ❌ Error descargando {filename}: {e}")
    
    return downloaded_count

# =============================================================================
# EJECUTAR DESCARGAS
# =============================================================================
print("\n🚀 Iniciando descarga de imágenes...")

# Descargar desde diferentes fuentes
unsplash_count = download_from_unsplash()
pixabay_count = download_from_pixabay()
synthetic_count = create_synthetic_images()

total_downloaded = unsplash_count + pixabay_count + synthetic_count

print(f"\n📊 RESUMEN DE DESCARGAS:")
print(f"   📥 Unsplash: {unsplash_count} imágenes")
print(f"   📥 Pixabay: {pixabay_count} imágenes")
print(f"   🎨 Sintéticas: {synthetic_count} imágenes")
print(f"   📊 Total: {total_downloaded} imágenes")

# =============================================================================
# VERIFICAR RESULTADO FINAL
# =============================================================================
print(f"\n📁 Verificando directorio: {IMAGES_DIR}")

total_images = 0
for class_name in classes:
    class_dir = os.path.join(IMAGES_DIR, class_name)
    if os.path.exists(class_dir):
        images_in_class = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"   📂 {class_name}: {images_in_class} imágenes")
        total_images += images_in_class

print(f"\n✅ Total de imágenes disponibles: {total_images}")
print(f"🎯 Objetivo: 100 imágenes (10 por clase)")

if total_images >= 100:
    print("🎉 ¡Objetivo alcanzado! Tienes suficientes imágenes para probar los modelos.")
else:
    print(f"⚠️  Faltan {100 - total_images} imágenes. Considera descargar más manualmente.")

print(f"\n📂 Las imágenes están en: {os.path.abspath(IMAGES_DIR)}")
print("🚀 ¡Listo para probar los modelos!")
