# 🚗 Datasets de Autos Alternativos

## 📚 Datasets Disponibles en TensorFlow Datasets

### **1. Cars196 (Stanford)**
- **URL**: https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz
- **Estado**: ❌ **INACCESIBLE** (Error 404)
- **Alternativas**: Ver opciones abajo

### **2. Datasets Alternativos Funcionales**

#### **A. CIFAR-10 (Para Pruebas)**
```python
# Cargar CIFAR-10 como alternativa
dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)
```
- **Clases**: 10 (incluye automóviles)
- **Tamaño**: ~170 MB
- **Ventaja**: Siempre disponible, rápido de descargar

#### **B. ImageNet (Subconjunto)**
```python
# Cargar subconjunto de ImageNet
dataset, info = tfds.load('imagenet2012_subset', with_info=True)
```
- **Clases**: 1000 (incluye muchas clases de vehículos)
- **Tamaño**: ~150 GB
- **Ventaja**: Dataset muy completo

#### **C. Oxford-IIIT Pet**
```python
# Dataset de mascotas (para transfer learning)
dataset, info = tfds.load('oxford_iiit_pet', with_info=True)
```
- **Clases**: 37
- **Tamaño**: ~800 MB
- **Ventaja**: Bueno para practicar transfer learning

## 🌐 Fuentes Externas de Datasets

### **1. Kaggle Datasets**
- **Car Images Dataset**: https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set
- **Vehicle Dataset**: https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set
- **Car Brand Classification**: https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

### **2. Roboflow Universe**
- **Vehicle Detection**: https://universe.roboflow.com/roboflow-universe-projects/vehicle-detection-3zvbc
- **Car Classification**: https://universe.roboflow.com/roboflow-universe-projects/car-classification-1

### **3. GitHub Repositories**
- **Cars Dataset**: https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data
- **Vehicle Classification**: https://github.com/udacity/CarND-Vehicle-Detection

## 🔧 Soluciones para el Problema Actual

### **Opción 1: Usar CIFAR-10 (Recomendado para Pruebas)**
```python
# Código para usar CIFAR-10
import tensorflow_datasets as tfds

# Cargar CIFAR-10
dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Filtrar solo imágenes de autos (clase 1)
def filter_cars(image, label):
    return label == 1  # Clase 1 = automóviles

train_cars = train_dataset.filter(filter_cars)
test_cars = test_dataset.filter(filter_cars)
```

### **Opción 2: Dataset Sintético (Para Demostración)**
```python
# Crear dataset sintético de autos
import numpy as np
import cv2

def crear_autos_sinteticos(num_classes=10, samples_per_class=100):
    images = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            # Crear imagen de "auto" sintético
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Dibujar forma de auto
            color = [(class_id * 25) % 255, (class_id * 50) % 255, (class_id * 75) % 255]
            cv2.rectangle(img, (50, 100), (174, 150), color, -1)  # Cuerpo del auto
            cv2.rectangle(img, (60, 80), (164, 100), color, -1)   # Techo del auto
            cv2.circle(img, (80, 160), 15, (0, 0, 0), -1)        # Rueda 1
            cv2.circle(img, (144, 160), 15, (0, 0, 0), -1)       # Rueda 2
            
            images.append(img)
            labels.append(class_id)
    
    return np.array(images), np.array(labels)
```

### **Opción 3: Descargar Manualmente**
```python
# Descargar dataset manualmente
import urllib.request
import zipfile
import os

def descargar_dataset_manual():
    # URLs alternativas para Cars196
    urls = [
        "https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/raw/master/picture-dataset.zip",
        "https://storage.googleapis.com/download.tensorflow.org/data/cars196.zip"
    ]
    
    for url in urls:
        try:
            print(f"Intentando descargar desde: {url}")
            urllib.request.urlretrieve(url, "cars_dataset.zip")
            
            with zipfile.ZipFile("cars_dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("cars_dataset")
            
            print("✅ Dataset descargado exitosamente")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    return False
```

## 📊 Comparación de Datasets

| Dataset | Clases | Tamaño | Disponibilidad | Velocidad |
|---------|--------|--------|----------------|-----------|
| Cars196 | 196 | 1.82 GB | ❌ Inaccesible | - |
| CIFAR-10 | 10 | 170 MB | ✅ Disponible | ⚡ Rápido |
| ImageNet | 1000 | 150 GB | ✅ Disponible | 🐌 Lento |
| Sintético | 10 | 50 MB | ✅ Siempre | ⚡ Muy rápido |

## 🎯 Recomendaciones

### **Para Pruebas Rápidas:**
1. **CIFAR-10** - Dataset pequeño y confiable
2. **Dataset Sintético** - Para demostración inmediata

### **Para Proyecto Completo:**
1. **Kaggle Datasets** - Descargar manualmente
2. **ImageNet** - Dataset completo pero lento
3. **Roboflow** - Datasets especializados

### **Para Transfer Learning:**
1. **Pre-entrenar en ImageNet**
2. **Fine-tune en dataset específico**
3. **Usar ResNet50V2 pre-entrenado**

## 🚀 Código de Solución Rápida

```python
# Solución inmediata para Google Colab
import tensorflow_datasets as tfds
import tensorflow as tf

# Opción 1: CIFAR-10 (automóviles)
dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# Filtrar solo autos
def filter_cars(image, label):
    return label == 1  # Clase 1 = automóviles

train_cars = train_dataset.filter(filter_cars)
test_cars = test_dataset.filter(filter_cars)

print(f"Autos en entrenamiento: {len(list(train_cars.as_numpy_iterator()))}")
print(f"Autos en prueba: {len(list(test_cars.as_numpy_iterator()))}")
```

## 📝 Notas Importantes

1. **Cars196** tiene problemas de disponibilidad frecuentes
2. **CIFAR-10** es una excelente alternativa para pruebas
3. **Dataset sintético** es perfecto para demostración
4. **Kaggle** tiene datasets de autos actualizados
5. **Transfer learning** funciona bien con cualquier dataset

---

**Consejo**: Para tu proyecto, recomiendo empezar con CIFAR-10 o el dataset sintético para probar el código, y luego migrar a un dataset más específico de autos cuando esté disponible.
