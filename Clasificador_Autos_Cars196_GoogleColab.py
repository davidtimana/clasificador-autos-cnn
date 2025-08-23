# -*- coding: utf-8 -*-
"""
CLASIFICADOR DE AUTOS CNN - DATASET CARS196
Google Colab Version
============================================

Este script está optimizado para ejecutarse en Google Colab.
Incluye todas las funcionalidades del clasificador de autos:
- Transfer Learning con ResNet50V2
- Data Augmentation
- Fine-tuning
- Evaluación completa
- Soporte para imágenes personalizadas

Autor: David Timana
Fecha: 2024
Curso: Visión por Computador
"""

# =============================================================================
# CONFIGURACIÓN INICIAL PARA GOOGLE COLAB
# =============================================================================
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Configuración para Google Colab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# INSTALACIÓN DE DEPENDENCIAS (Google Colab)
# =============================================================================
print("🔧 Instalando dependencias...")

# Instalar tensorflow-datasets si no está disponible
try:
    import tensorflow_datasets as tfds
except ImportError:
    print("Instalando tensorflow-datasets...")
    !pip install tensorflow-datasets

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import glob
import zipfile
from google.colab import files
from google.colab import drive

print(f"✅ TensorFlow {tf.__version__} cargado exitosamente")
print(f"🚀 GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# CONFIGURACIÓN DE PARÁMETROS
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 196
TRAIN_SPLIT = 0.8
VALIDATION_SPLIT = 0.2

print(f"📊 Configuración:")
print(f"   🖼️  Tamaño de imagen: {IMG_SIZE}x{IMG_SIZE}")
print(f"   📦 Batch size: {BATCH_SIZE}")
print(f"   🔄 Épocas: {EPOCHS}")
print(f"   🎯 Learning rate: {LEARNING_RATE}")
print(f"   🏷️  Clases: {NUM_CLASSES}")

# =============================================================================
# FUNCIÓN 1: CARGA DEL DATASET CARS196
# =============================================================================
def cargar_dataset_cars196():
    """
    Carga el dataset Cars196 de TensorFlow Datasets
    """
    print("\n🔄 Cargando dataset Cars196...")
    
    try:
        # Cargar dataset
        dataset, info = tfds.load('cars196', 
                                 with_info=True, 
                                 as_supervised=True,
                                 split=['train', 'test'])
        
        train_dataset, test_dataset = dataset[0], dataset[1]
        
        print(f"✅ Dataset cargado exitosamente:")
        print(f"   📊 Entrenamiento: {info.splits['train'].num_examples:,} imágenes")
        print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
        print(f"   🏷️  Clases: {info.features['label'].num_classes}")
        print(f"   📁 Tamaño total: ~1.82 GB")
        
        return train_dataset, test_dataset, info
        
    except Exception as e:
        print(f"❌ Error al cargar dataset: {e}")
        return None, None, None

# =============================================================================
# FUNCIÓN 2: PREPROCESAMIENTO DE IMÁGENES
# =============================================================================
def preprocesar_imagen(image, label):
    """
    Preprocesa las imágenes para el entrenamiento con data augmentation
    """
    # Redimensionar imagen
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Normalizar valores de píxeles (0-1)
    image = tf.cast(image, tf.float32) / 255.0
    
    # Data augmentation para entrenamiento
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_flip_left_right(image)
    
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, 0.2)
    
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image, label

def preprocesar_imagen_test(image, label):
    """
    Preprocesa las imágenes para validación/prueba (sin augmentation)
    """
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# =============================================================================
# FUNCIÓN 3: CREACIÓN DEL MODELO CON TRANSFER LEARNING
# =============================================================================
def crear_modelo_transfer_learning():
    """
    Crea un modelo usando transfer learning con ResNet50V2
    """
    print("\n🚀 Creando modelo con Transfer Learning (ResNet50V2)...")
    
    # Cargar modelo base pre-entrenado
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Congelar las capas del modelo base
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
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print("✅ Modelo de transfer learning creado exitosamente")
    print(f"   📊 Parámetros totales: {model.count_params():,}")
    
    return model, base_model

# =============================================================================
# FUNCIÓN 4: ENTRENAMIENTO DEL MODELO
# =============================================================================
def entrenar_modelo(model, train_dataset, val_dataset, base_model=None):
    """
    Entrena el modelo con fine-tuning
    """
    print("\n🚀 Iniciando entrenamiento...")
    
    # Compilar modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_5_accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'mejor_modelo_autos.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Entrenamiento inicial
    print("📚 Fase 1: Entrenamiento inicial...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning si hay modelo base
    if base_model is not None:
        print("\n🔧 Fase 2: Fine-tuning...")
        
        # Descongelar algunas capas del modelo base
        base_model.trainable = True
        
        # Congelar las primeras capas
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Recompilar con learning rate más bajo
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE/10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        # Fine-tuning
        history_fine = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combinar historiales
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    print("✅ Entrenamiento completado exitosamente")
    return history

# =============================================================================
# FUNCIÓN 5: EVALUACIÓN DEL MODELO
# =============================================================================
def evaluar_modelo(model, test_dataset, info):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    print("\n📊 Evaluando modelo...")
    
    # Evaluación
    test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_dataset, verbose=1)
    
    print(f"\n📈 Resultados de evaluación:")
    print(f"   📉 Pérdida de prueba: {test_loss:.4f}")
    print(f"   ✅ Precisión de prueba: {test_accuracy:.4f}")
    print(f"   🏆 Top-5 precisión: {test_top5_accuracy:.4f}")
    
    return test_accuracy, test_top5_accuracy

# =============================================================================
# FUNCIÓN 6: VISUALIZACIÓN DE RESULTADOS
# =============================================================================
def visualizar_resultados(history):
    """
    Visualiza los resultados del entrenamiento
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precisión
    axes[0, 0].plot(history.history['accuracy'], label='Entrenamiento')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validación')
    axes[0, 0].set_title('Precisión del Modelo')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Precisión')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Pérdida
    axes[0, 1].plot(history.history['loss'], label='Entrenamiento')
    axes[0, 1].plot(history.history['val_loss'], label='Validación')
    axes[0, 1].set_title('Pérdida del Modelo')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Pérdida')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Top-5 Precisión
    axes[1, 0].plot(history.history['top_5_accuracy'], label='Entrenamiento')
    axes[1, 0].plot(history.history['val_top_5_accuracy'], label='Validación')
    axes[1, 0].set_title('Top-5 Precisión del Modelo')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Top-5 Precisión')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# FUNCIÓN 7: PREDICCIÓN DE IMÁGENES PERSONALIZADAS
# =============================================================================
def predecir_imagen_personalizada(model, ruta_imagen):
    """
    Predice la clase de una imagen personalizada
    """
    # Cargar y preprocesar imagen
    img = load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    # Predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Top-5 predicciones
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_confidences = predictions[0][top_5_indices]
    
    return predicted_class, confidence, top_5_indices, top_5_confidences

# =============================================================================
# FUNCIÓN 8: SUBIR IMÁGENES PERSONALIZADAS (Google Colab)
# =============================================================================
def subir_imagenes_personalizadas():
    """
    Permite subir imágenes personalizadas en Google Colab
    """
    print("\n📁 Subiendo imágenes personalizadas...")
    
    # Crear carpeta para imágenes
    if not os.path.exists('imagenes_personalizadas'):
        os.makedirs('imagenes_personalizadas')
    
    # Subir archivo ZIP con imágenes
    print("📤 Por favor, sube un archivo ZIP con tus imágenes:")
    uploaded = files.upload()
    
    # Extraer imágenes
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('imagenes_personalizadas')
            print(f"✅ Imágenes extraídas de {filename}")
    
    # Listar imágenes
    imagenes = glob.glob('imagenes_personalizadas/**/*.jpg', recursive=True)
    imagenes.extend(glob.glob('imagenes_personalizadas/**/*.jpeg', recursive=True))
    imagenes.extend(glob.glob('imagenes_personalizadas/**/*.png', recursive=True))
    
    print(f"📊 Encontradas {len(imagenes)} imágenes")
    return imagenes

# =============================================================================
# FUNCIÓN 9: EVALUAR IMÁGENES PERSONALIZADAS
# =============================================================================
def evaluar_imagenes_personalizadas(model, imagenes):
    """
    Evalúa las imágenes personalizadas
    """
    print(f"\n🔍 Evaluando {len(imagenes)} imágenes personalizadas...")
    
    resultados = []
    
    for i, ruta_imagen in enumerate(imagenes):
        try:
            clase_predicha, confianza, top5_indices, top5_confianzas = predecir_imagen_personalizada(model, ruta_imagen)
            
            resultado = {
                'imagen': os.path.basename(ruta_imagen),
                'clase_predicha': clase_predicha,
                'confianza': confianza,
                'top5_clases': top5_indices,
                'top5_confianzas': top5_confianzas
            }
            resultados.append(resultado)
            
            print(f"   {i+1:3d}. {os.path.basename(ruta_imagen):30s} | Clase: {clase_predicha:3d} | Conf: {confianza:.3f}")
            
        except Exception as e:
            print(f"   ❌ Error en {os.path.basename(ruta_imagen)}: {e}")
    
    return resultados

# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================
def main():
    """
    Función principal del clasificador
    """
    print("=" * 60)
    print("🚗 CLASIFICADOR DE AUTOS CNN - DATASET CARS196")
    print("📚 Google Colab Version - Visión por Computador")
    print("=" * 60)
    
    # 1. Cargar dataset
    train_dataset, test_dataset, info = cargar_dataset_cars196()
    if train_dataset is None:
        print("❌ No se pudo cargar el dataset")
        return
    
    # 2. Preprocesar datos
    print("\n🔄 Preprocesando datos...")
    
    # Aplicar preprocesamiento
    train_dataset = train_dataset.map(preprocesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(preprocesar_imagen_test, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Configurar para rendimiento
    train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # Dividir train en train y validation
    train_size = int(TRAIN_SPLIT * info.splits['train'].num_examples)
    val_size = info.splits['train'].num_examples - train_size
    
    train_dataset_final = train_dataset.take(train_size // BATCH_SIZE)
    val_dataset = train_dataset.skip(train_size // BATCH_SIZE).take(val_size // BATCH_SIZE)
    
    print(f"✅ Datos preprocesados:")
    print(f"   📊 Entrenamiento: {train_size:,} imágenes")
    print(f"   📊 Validación: {val_size:,} imágenes")
    print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
    
    # 3. Crear modelo
    model, base_model = crear_modelo_transfer_learning()
    model.summary()
    
    # 4. Entrenar modelo
    history = entrenar_modelo(model, train_dataset_final, val_dataset, base_model)
    if history is None:
        print("❌ Error durante el entrenamiento")
        return
    
    # 5. Evaluar modelo
    test_accuracy, test_top5_accuracy = evaluar_modelo(model, test_dataset, info)
    
    # 6. Visualizar resultados
    visualizar_resultados(history)
    
    # 7. Guardar modelo
    model.save('modelo_autos_final.h5')
    print("\n💾 Modelo guardado como 'modelo_autos_final.h5'")
    
    # 8. Resumen final
    print(f"\n" + "=" * 60)
    print(f"🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print(f"=" * 60)
    print(f"📊 Precisión final: {test_accuracy:.4f}")
    print(f"🏆 Top-5 precisión: {test_top5_accuracy:.4f}")
    print(f"🎯 Modelo listo para usar con imágenes personalizadas!")
    print(f"📁 Archivos generados:")
    print(f"   - modelo_autos_final.h5 (modelo final)")
    print(f"   - mejor_modelo_autos.h5 (mejor modelo durante entrenamiento)")
    
    # 9. Opción para evaluar imágenes personalizadas
    print(f"\n" + "=" * 60)
    print(f"🔍 EVALUACIÓN DE IMÁGENES PERSONALIZADAS")
    print(f"=" * 60)
    
    evaluar_personalizadas = input("¿Deseas evaluar imágenes personalizadas? (s/n): ").lower().strip()
    
    if evaluar_personalizadas == 's':
        try:
            imagenes = subir_imagenes_personalizadas()
            if imagenes:
                resultados = evaluar_imagenes_personalizadas(model, imagenes)
                
                # Crear DataFrame con resultados
                df_resultados = pd.DataFrame([
                    {
                        'Imagen': r['imagen'],
                        'Clase_Predicha': r['clase_predicha'],
                        'Confianza': r['confianza'],
                        'Top1_Clase': r['top5_clases'][0],
                        'Top1_Confianza': r['top5_confianzas'][0]
                    }
                    for r in resultados
                ])
                
                print(f"\n📊 Resumen de predicciones:")
                print(df_resultados.head(10))
                
                # Guardar resultados
                df_resultados.to_csv('resultados_imagenes_personalizadas.csv', index=False)
                print(f"\n💾 Resultados guardados en 'resultados_imagenes_personalizadas.csv'")
                
        except Exception as e:
            print(f"❌ Error al evaluar imágenes personalizadas: {e}")
    
    print(f"\n🎉 ¡Proyecto completado exitosamente!")

# =============================================================================
# EJECUCIÓN DEL PROGRAMA
# =============================================================================
if __name__ == "__main__":
    main()
