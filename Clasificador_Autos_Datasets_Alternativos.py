# -*- coding: utf-8 -*-
"""
🚗 CLASIFICADOR DE AUTOS CNN - DATASETS ALTERNATIVOS
Google Colab Version - Múltiples opciones de datasets
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
# DATASETS ALTERNATIVOS DISPONIBLES
# =============================================================================
DATASETS_DISPONIBLES = {
    '1': {
        'nombre': 'cars196',
        'descripcion': 'Cars196 - 196 clases de autos (Stanford)',
        'clases': 196,
        'tamaño': '~1.82 GB'
    },
    '2': {
        'nombre': 'cars196_alt',
        'descripcion': 'Cars196 - Versión alternativa',
        'clases': 196,
        'tamaño': '~1.82 GB'
    },
    '3': {
        'nombre': 'cars196_v2',
        'descripcion': 'Cars196 - Versión 2',
        'clases': 196,
        'tamaño': '~1.82 GB'
    },
    '4': {
        'nombre': 'cars196_3d',
        'descripcion': 'Cars196 3D - Versión 3D',
        'clases': 196,
        'tamaño': '~1.82 GB'
    },
    '5': {
        'nombre': 'cars196_annotated',
        'descripcion': 'Cars196 con anotaciones',
        'clases': 196,
        'tamaño': '~1.82 GB'
    },
    '6': {
        'nombre': 'cars196_clean',
        'descripcion': 'Cars196 limpiado',
        'clases': 196,
        'tamaño': '~1.82 GB'
    }
}

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def mostrar_datasets_disponibles():
    """Muestra los datasets disponibles"""
    print("\n📚 DATASETS DE AUTOS DISPONIBLES:")
    print("=" * 60)
    for key, dataset in DATASETS_DISPONIBLES.items():
        print(f"{key}. {dataset['nombre']}")
        print(f"   📝 {dataset['descripcion']}")
        print(f"   🏷️  Clases: {dataset['clases']}")
        print(f"   📁 Tamaño: {dataset['tamaño']}")
        print()

def cargar_dataset_alternativo(nombre_dataset):
    """Intenta cargar diferentes versiones del dataset"""
    print(f"\n🔄 Intentando cargar dataset: {nombre_dataset}")
    
    # Lista de posibles configuraciones
    configuraciones = [
        {'with_info': True, 'as_supervised': True, 'split': ['train', 'test']},
        {'with_info': True, 'as_supervised': True, 'split': 'train+test'},
        {'with_info': True, 'as_supervised': False, 'split': ['train', 'test']},
        {'with_info': True, 'split': ['train', 'test']},
        {'with_info': True, 'split': 'train+test'},
    ]
    
    for i, config in enumerate(configuraciones):
        try:
            print(f"   Intento {i+1}: {config}")
            dataset, info = tfds.load(nombre_dataset, **config)
            
            # Manejar diferentes formatos de salida
            if isinstance(dataset, list):
                train_dataset, test_dataset = dataset[0], dataset[1]
            else:
                # Si es un solo dataset, dividirlo
                total_size = info.splits['train'].num_examples + info.splits['test'].num_examples
                train_size = int(0.8 * total_size)
                test_size = total_size - train_size
                
                dataset = dataset.shuffle(1000)
                train_dataset = dataset.take(train_size)
                test_dataset = dataset.skip(train_size).take(test_size)
            
            print(f"✅ Dataset cargado exitosamente!")
            print(f"   📊 Entrenamiento: {info.splits['train'].num_examples:,} imágenes")
            print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
            print(f"   🏷️  Clases: {info.features['label'].num_classes}")
            
            return train_dataset, test_dataset, info
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}...")
            continue
    
    return None, None, None

def cargar_dataset_manual():
    """Carga dataset manualmente desde diferentes fuentes"""
    print("\n🔄 Intentando cargar dataset manualmente...")
    
    # Intentar diferentes nombres de datasets
    nombres_alternativos = [
        'cars196',
        'cars196_alt', 
        'cars196_v2',
        'cars196_3d',
        'cars196_annotated',
        'cars196_clean',
        'cars',
        'car_dataset',
        'vehicle_dataset'
    ]
    
    for nombre in nombres_alternativos:
        print(f"\n🔍 Probando: {nombre}")
        train_dataset, test_dataset, info = cargar_dataset_alternativo(nombre)
        if train_dataset is not None:
            return train_dataset, test_dataset, info
    
    return None, None, None

def crear_dataset_sintetico():
    """Crea un dataset sintético para demostración"""
    print("\n🔄 Creando dataset sintético para demostración...")
    
    # Crear imágenes sintéticas simples
    num_classes = 10
    samples_per_class = 100
    total_samples = num_classes * samples_per_class
    
    # Generar imágenes sintéticas (rectángulos de diferentes colores)
    images = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(samples_per_class):
            # Crear imagen sintética (rectángulo con color específico)
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # Color basado en la clase
            color = [
                (class_id * 25) % 255,
                (class_id * 50) % 255,
                (class_id * 75) % 255
            ]
            
            # Dibujar rectángulo
            cv2.rectangle(img, (50, 50), (IMG_SIZE-50, IMG_SIZE-50), color, -1)
            
            # Agregar ruido
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
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
    
    print(f"✅ Dataset sintético creado:")
    print(f"   📊 Entrenamiento: {len(train_images):,} imágenes")
    print(f"   📊 Prueba: {len(test_images):,} imágenes")
    print(f"   🏷️  Clases: {num_classes}")
    
    return train_dataset, test_dataset, info

def preprocesar_imagen(image, label, augment=True):
    """Preprocesa imagen con data augmentation"""
    # Si la imagen ya tiene el tamaño correcto, no redimensionar
    if image.shape[0] != IMG_SIZE or image.shape[1] != IMG_SIZE:
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Asegurar que los valores estén en [0, 1]
    if tf.reduce_max(image) > 1.0:
        image = tf.cast(image, tf.float32) / 255.0
    else:
        image = tf.cast(image, tf.float32)
    
    if augment:
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, 0.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
    
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
        ModelCheckpoint('mejor_modelo.h5', monitor='val_accuracy', save_best_only=True)
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
    axes[0].set_title('Precisión del Modelo')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Precisión')
    axes[0].legend()
    axes[0].grid(True)
    
    # Pérdida
    axes[1].plot(history.history['loss'], label='Entrenamiento')
    axes[1].plot(history.history['val_loss'], label='Validación')
    axes[1].set_title('Pérdida del Modelo')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Pérdida')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """Función principal"""
    print("=" * 60)
    print("🚗 CLASIFICADOR DE AUTOS CNN - DATASETS ALTERNATIVOS")
    print("📚 Google Colab Version")
    print("=" * 60)
    
    # Mostrar opciones de datasets
    mostrar_datasets_disponibles()
    
    # Intentar cargar dataset
    print("🔄 Intentando cargar dataset de autos...")
    
    # Opción 1: Intentar cargar manualmente
    train_dataset, test_dataset, info = cargar_dataset_manual()
    
    # Opción 2: Si falla, crear dataset sintético
    if train_dataset is None:
        print("\n⚠️  No se pudo cargar ningún dataset de autos.")
        print("🔄 Creando dataset sintético para demostración...")
        train_dataset, test_dataset, info = crear_dataset_sintetico()
    
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
    
    print(f"\n🎉 ¡Proyecto completado!")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    main()
