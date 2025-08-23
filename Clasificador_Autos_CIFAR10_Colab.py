# -*- coding: utf-8 -*-
"""
🚗 CLASIFICADOR DE AUTOS CNN - CIFAR-10 ALTERNATIVA
Google Colab Version - Usando CIFAR-10 como dataset de autos
Autor: David Timana | Curso: Visión por Computador
"""

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# INSTALACIÓN E IMPORTACIÓN
# =============================================================================
print("🔧 Instalando dependencias...")
try:
    import tensorflow_datasets as tfds
except ImportError:
    !pip install tensorflow-datasets

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
import zipfile
from google.colab import files
import cv2 # Added missing import for cv2

print(f"✅ TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# PARÁMETROS
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

print(f"📊 Config: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def cargar_cifar10_autos():
    """Carga CIFAR-10 y filtra solo imágenes de autos"""
    print("\n🔄 Cargando CIFAR-10 dataset...")
    
    # Cargar CIFAR-10 completo
    dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    
    print(f"✅ CIFAR-10 cargado:")
    print(f"   📊 Total entrenamiento: {info.splits['train'].num_examples:,} imágenes")
    print(f"   📊 Total prueba: {info.splits['test'].num_examples:,} imágenes")
    print(f"   🏷️  Clases: {info.features['label'].num_classes}")
    
    # Filtrar solo autos (clase 1 en CIFAR-10)
    print("\n🚗 Filtrando solo imágenes de automóviles...")
    
    def filter_cars(image, label):
        return label == 1  # Clase 1 = automóviles en CIFAR-10
    
    train_cars = train_dataset.filter(filter_cars)
    test_cars = test_dataset.filter(filter_cars)
    
    # Contar imágenes de autos
    train_cars_count = len(list(train_cars.as_numpy_iterator()))
    test_cars_count = len(list(test_cars.as_numpy_iterator()))
    
    print(f"✅ Automóviles encontrados:")
    print(f"   📊 Entrenamiento: {train_cars_count:,} autos")
    print(f"   📊 Prueba: {test_cars_count:,} autos")
    
    # Crear info personalizado para autos
    class CarInfo:
        def __init__(self, train_count, test_count):
            self.splits = {
                'train': type('obj', (object,), {'num_examples': train_count}),
                'test': type('obj', (object,), {'num_examples': test_count})
            }
            self.features = type('obj', (object,), {'label': type('obj', (object,), {'num_classes': 1})})
    
    info_cars = CarInfo(train_cars_count, test_cars_count)
    
    return train_cars, test_cars, info_cars

def crear_dataset_sintetico_autos():
    """Crea un dataset sintético de autos más realista"""
    print("\n🔄 Creando dataset sintético de autos...")
    
    num_classes = 5  # 5 tipos diferentes de autos
    samples_per_class = 200
    total_samples = num_classes * samples_per_class
    
    # Generar imágenes sintéticas de autos
    images = []
    labels = []
    
    # Colores de autos comunes
    car_colors = [
        [255, 0, 0],    # Rojo
        [0, 0, 255],    # Azul
        [0, 255, 0],    # Verde
        [255, 255, 0],  # Amarillo
        [128, 128, 128] # Gris
    ]
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            # Crear imagen de auto sintético
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # Color del auto
            color = car_colors[class_id]
            
            # Dibujar auto más realista
            # Cuerpo principal del auto
            cv2.rectangle(img, (60, 120), (164, 180), color, -1)
            
            # Techo del auto
            cv2.rectangle(img, (70, 100), (154, 120), color, -1)
            
            # Ventanas (más oscuras)
            window_color = [max(0, c - 100) for c in color]
            cv2.rectangle(img, (75, 105), (149, 115), window_color, -1)
            
            # Ruedas
            cv2.circle(img, (85, 190), 20, (0, 0, 0), -1)
            cv2.circle(img, (139, 190), 20, (0, 0, 0), -1)
            
            # Detalles del auto (faros, parachoques, etc.)
            cv2.rectangle(img, (65, 175), (159, 185), [max(0, c - 50) for c in color], -1)
            
            # Agregar ruido para realismo
            noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
            img = np.clip(img + noise, 0, 255)
            
            images.append(img)
            labels.append(class_id)
    
    # Convertir a arrays
    images = np.array(images, dtype=np.float32) / 255.0
    labels = np.array(labels)
    
    # Dividir en train/test
    train_size = int(0.8 * len(images))
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]
    
    # Crear datasets de TensorFlow
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    # Crear info sintético
    class Info:
        def __init__(self):
            self.splits = {
                'train': type('obj', (object,), {'num_examples': len(train_images)}),
                'test': type('obj', (object,), {'num_examples': len(test_images)})
            }
            self.features = type('obj', (object,), {'label': type('obj', (object,), {'num_classes': num_classes})})
    
    info = Info()
    
    print(f"✅ Dataset sintético de autos creado:")
    print(f"   📊 Entrenamiento: {len(train_images):,} autos")
    print(f"   📊 Prueba: {len(test_images):,} autos")
    print(f"   🏷️  Clases: {num_classes} (diferentes colores/tipos)")
    
    return train_dataset, test_dataset, info

def preprocesar_imagen(image, label, augment=True):
    """Preprocesa imagen con data augmentation"""
    # Redimensionar a IMG_SIZE
    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalizar valores
    if tf.reduce_max(image) > 1.0:
        image = tf.cast(image, tf.float32) / 255.0
    else:
        image = tf.cast(image, tf.float32)
    
    # Data augmentation
    if augment:
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, 0.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_hue(image, 0.1)
    
    return image, label

def crear_modelo(num_classes):
    """Crea modelo con transfer learning"""
    print(f"\n🚀 Creando modelo ResNet50V2 para {num_classes} clases...")
    
    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"✅ Modelo creado: {model.count_params():,} parámetros")
    return model, base_model

def entrenar_modelo(model, train_dataset, val_dataset, base_model=None):
    """Entrena el modelo con fine-tuning"""
    print("\n🚀 Iniciando entrenamiento...")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint('mejor_modelo_autos.h5', monitor='val_accuracy', save_best_only=True)
    ]
    
    # Entrenamiento inicial
    print("📚 Fase 1: Entrenamiento inicial...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)
    
    # Fine-tuning
    if base_model is not None:
        print("\n🔧 Fase 2: Fine-tuning...")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_fine = model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks)
        
        # Combinar historiales
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    return history

def evaluar_modelo(model, test_dataset):
    """Evalúa el modelo"""
    print("\n📊 Evaluando modelo...")
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    
    print(f"\n📈 Resultados:")
    print(f"   📉 Pérdida: {test_loss:.4f}")
    print(f"   ✅ Precisión: {test_accuracy:.4f}")
    
    return test_accuracy

def visualizar_resultados(history):
    """Visualiza resultados del entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Precisión
    axes[0].plot(history.history['accuracy'], label='Entrenamiento')
    axes[0].plot(history.history['val_accuracy'], label='Validación')
    axes[0].set_title('Precisión del Modelo de Autos')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión')
    axes[0].legend()
    axes[0].grid(True)
    
    # Pérdida
    axes[1].plot(history.history['loss'], label='Entrenamiento')
    axes[1].plot(history.history['val_loss'], label='Validación')
    axes[1].set_title('Pérdida del Modelo de Autos')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Pérdida')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def mostrar_ejemplos_dataset(dataset, num_ejemplos=5):
    """Muestra ejemplos del dataset"""
    print(f"\n🖼️  Ejemplos del dataset:")
    
    fig, axes = plt.subplots(1, num_ejemplos, figsize=(15, 3))
    
    for i, (image, label) in enumerate(dataset.take(num_ejemplos)):
        if num_ejemplos == 1:
            ax = axes
        else:
            ax = axes[i]
        
        # Redimensionar si es necesario
        if image.shape[0] != IMG_SIZE:
            image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
        
        ax.imshow(image)
        ax.set_title(f'Clase: {label.numpy()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """Función principal"""
    print("=" * 60)
    print("🚗 CLASIFICADOR DE AUTOS CNN - CIFAR-10 ALTERNATIVA")
    print("📚 Google Colab Version")
    print("=" * 60)
    
    # Preguntar qué dataset usar
    print("\n🎯 Selecciona el dataset a usar:")
    print("1. CIFAR-10 (automóviles reales)")
    print("2. Dataset sintético (autos generados)")
    
    opcion = input("Elige una opción (1 o 2): ").strip()
    
    if opcion == "1":
        print("\n🚗 Usando CIFAR-10 (automóviles reales)...")
        train_dataset, test_dataset, info = cargar_cifar10_autos()
    else:
        print("\n🎨 Usando dataset sintético...")
        train_dataset, test_dataset, info = crear_dataset_sintetico_autos()
    
    # Mostrar ejemplos
    mostrar_ejemplos_dataset(train_dataset)
    
    # Preprocesar datos
    print("\n🔄 Preprocesando datos...")
    train_dataset = train_dataset.map(lambda x, y: preprocesar_imagen(x, y, augment=True), 
                                     num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(lambda x, y: preprocesar_imagen(x, y, augment=False), 
                                   num_parallel_calls=tf.data.AUTOTUNE)
    
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Dividir train/validation
    train_size = int(0.8 * info.splits['train'].num_examples)
    val_size = info.splits['train'].num_examples - train_size
    
    train_dataset_final = train_dataset.take(train_size // BATCH_SIZE)
    val_dataset = train_dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    
    print(f"✅ Datos: {train_size:,} train, {val_size:,} val, {info.splits['test'].num_examples:,} test")
    
    # Crear modelo
    num_classes = info.features.label.num_classes
    model, base_model = crear_modelo(num_classes)
    model.summary()
    
    # Entrenar modelo
    history = entrenar_modelo(model, train_dataset_final, val_dataset, base_model)
    
    # Evaluar modelo
    test_accuracy = evaluar_modelo(model, test_dataset)
    
    # Visualizar resultados
    visualizar_resultados(history)
    
    # Guardar modelo
    model.save('modelo_autos_final.h5')
    print("\n💾 Modelo guardado: modelo_autos_final.h5")
    
    # Resumen final
    print(f"\n" + "=" * 60)
    print(f"🎉 ENTRENAMIENTO COMPLETADO")
    print(f"=" * 60)
    print(f"📊 Precisión: {test_accuracy:.4f}")
    print(f"🏷️  Clases: {num_classes}")
    
    if opcion == "1":
        print(f"📚 Dataset: CIFAR-10 (automóviles reales)")
    else:
        print(f"🎨 Dataset: Sintético (autos generados)")
    
    print(f"\n🎉 ¡Proyecto completado!")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    main()
