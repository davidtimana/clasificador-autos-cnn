# -*- coding: utf-8 -*-
"""
Clasificador de Autos CNN - Versión Simplificada para macOS
Dataset Cars196 - Visión por Computador

Esta es una versión simplificada del clasificador que evita problemas
comunes de TensorFlow en macOS. Incluye las funcionalidades principales
pero con configuraciones más estables.

Autor: David Timana
Fecha: 2024
"""

import os
import sys

# Configuración para evitar problemas en macOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reducir logs de TensorFlow
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Evitar conflictos de librerías

try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar TensorFlow: {e}")
    sys.exit(1)

import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 16  # Reducido para evitar problemas de memoria
EPOCHS = 10  # Reducido para prueba inicial
LEARNING_RATE = 0.001
NUM_CLASSES = 196

def cargar_dataset_cars196():
    """
    Carga el dataset Cars196 de TensorFlow Datasets
    """
    print("🔄 Cargando dataset Cars196...")
    
    try:
        # Cargar dataset con configuración simplificada
        dataset, info = tfds.load('cars196', 
                                 with_info=True, 
                                 as_supervised=True,
                                 split=['train', 'test'])
        
        train_dataset, test_dataset = dataset[0], dataset[1]
        
        print(f"✅ Dataset cargado exitosamente:")
        print(f"   📊 Entrenamiento: {info.splits['train'].num_examples:,} imágenes")
        print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
        print(f"   🏷️  Clases: {info.features['label'].num_classes}")
        
        return train_dataset, test_dataset, info
        
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return None, None, None

def preprocesar_imagen(image, label):
    """
    Preprocesa las imágenes para el entrenamiento
    """
    # Redimensionar imagen
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalizar valores de píxeles (0-1)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Data augmentation simple
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_flip_left_right(image)
    
    return image, label

def preprocesar_imagen_test(image, label):
    """
    Preprocesa las imágenes para validación/prueba
    """
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def crear_modelo_simple():
    """
    Crea un modelo CNN simple y estable
    """
    print("🏗️  Creando modelo CNN simple...")
    
    model = models.Sequential([
        # Primera capa convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Tercera capa convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten y capas densas
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print("✅ Modelo simple creado exitosamente")
    return model

def entrenar_modelo(model, train_dataset, val_dataset):
    """
    Entrena el modelo con configuración simplificada
    """
    print("🚀 Iniciando entrenamiento...")
    
    # Compilar modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks simples
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'mejor_modelo_simple.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenamiento
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Entrenamiento completado exitosamente")
        return history
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        return None

def evaluar_modelo(model, test_dataset):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    print("📊 Evaluando modelo...")
    
    try:
        # Evaluación
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
        
        print(f"\n📈 Resultados de evaluación:")
        print(f"   📉 Pérdida de prueba: {test_loss:.4f}")
        print(f"   ✅ Precisión de prueba: {test_accuracy:.4f}")
        
        return test_accuracy
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        return None

def main():
    """
    Función principal
    """
    print("=== CLASIFICADOR DE AUTOS CNN - VERSIÓN SIMPLIFICADA ===")
    print("Dataset Cars196 - Visión por Computador\n")
    
    # 1. Cargar dataset
    train_dataset, test_dataset, info = cargar_dataset_cars196()
    if train_dataset is None:
        print("❌ No se pudo cargar el dataset")
        return
    
    # 2. Preprocesar datos
    print("🔄 Preprocesando datos...")
    
    # Aplicar preprocesamiento
    train_dataset = train_dataset.map(preprocesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocesar_imagen_test, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Configurar para rendimiento
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Dividir train en train y validation
    train_size = int(0.8 * info.splits['train'].num_examples)
    val_size = info.splits['train'].num_examples - train_size
    
    train_dataset_final = train_dataset.take(train_size // BATCH_SIZE)
    val_dataset = train_dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    
    print(f"✅ Datos preprocesados:")
    print(f"   📊 Entrenamiento: {train_size:,} imágenes")
    print(f"   📊 Validación: {val_size:,} imágenes")
    print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
    
    # 3. Crear modelo
    model = crear_modelo_simple()
    model.summary()
    
    # 4. Entrenar modelo
    history = entrenar_modelo(model, train_dataset_final, val_dataset)
    if history is None:
        print("❌ Error durante el entrenamiento")
        return
    
    # 5. Evaluar modelo
    test_accuracy = evaluar_modelo(model, test_dataset)
    
    # 6. Guardar modelo
    try:
        model.save('modelo_autos_simple.h5')
        print("\n💾 Modelo guardado como 'modelo_autos_simple.h5'")
    except Exception as e:
        print(f"⚠️  Error al guardar modelo: {e}")
    
    # 7. Resumen final
    print(f"\n=== RESUMEN FINAL ===")
    if test_accuracy:
        print(f"✅ Precisión final: {test_accuracy:.4f}")
    print(f"🎯 Modelo listo para usar!")
    print(f"📁 Archivos generados:")
    print(f"   - modelo_autos_simple.h5 (modelo final)")
    print(f"   - mejor_modelo_simple.h5 (mejor modelo durante entrenamiento)")

if __name__ == "__main__":
    main()
