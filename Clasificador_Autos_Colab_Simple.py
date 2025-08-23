# -*- coding: utf-8 -*-
"""
🚗 CLASIFICADOR DE AUTOS CNN - GOOGLE COLAB
Dataset: Cars196 | Modelo: ResNet50V2 | Transfer Learning
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

print(f"✅ TensorFlow {tf.__version__} | GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# PARÁMETROS
# =============================================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 196

print(f"📊 Config: {IMG_SIZE}x{IMG_SIZE} | Batch: {BATCH_SIZE} | Epochs: {EPOCHS}")

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def cargar_dataset():
    """Carga Cars196 dataset"""
    print("\n🔄 Cargando Cars196 dataset...")
    dataset, info = tfds.load('cars196', with_info=True, as_supervised=True, split=['train', 'test'])
    train_dataset, test_dataset = dataset[0], dataset[1]
    
    print(f"✅ Dataset: {info.splits['train'].num_examples:,} train, {info.splits['test'].num_examples:,} test")
    return train_dataset, test_dataset, info

def preprocesar_imagen(image, label, augment=True):
    """Preprocesa imagen con data augmentation"""
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    
    if augment:
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_flip_left_right(image)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_brightness(image, 0.2)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image, label

def crear_modelo():
    """Crea modelo con transfer learning"""
    print("\n🚀 Creando modelo ResNet50V2...")
    
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
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print(f"✅ Modelo creado: {model.count_params():,} parámetros")
    return model, base_model

def entrenar_modelo(model, train_dataset, val_dataset, base_model=None):
    """Entrena el modelo con fine-tuning"""
    print("\n🚀 Iniciando entrenamiento...")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'top_5_accuracy']
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
            metrics=['accuracy', 'top_5_accuracy']
        )
        
        history_fine = model.fit(train_dataset, validation_data=val_dataset, epochs=10, callbacks=callbacks)
        
        # Combinar historiales
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    return history

def evaluar_modelo(model, test_dataset):
    """Evalúa el modelo"""
    print("\n📊 Evaluando modelo...")
    test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_dataset, verbose=1)
    
    print(f"\n📈 Resultados:")
    print(f"   📉 Pérdida: {test_loss:.4f}")
    print(f"   ✅ Precisión: {test_accuracy:.4f}")
    print(f"   🏆 Top-5: {test_top5_accuracy:.4f}")
    
    return test_accuracy, test_top5_accuracy

def visualizar_resultados(history):
    """Visualiza resultados del entrenamiento"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    # Top-5
    axes[2].plot(history.history['top_5_accuracy'], label='Entrenamiento')
    axes[2].plot(history.history['val_top_5_accuracy'], label='Validación')
    axes[2].set_title('Top-5 Precisión')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Top-5 Precisión')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def subir_imagenes_personalizadas():
    """Sube imágenes personalizadas"""
    print("\n📁 Subiendo imágenes personalizadas...")
    
    if not os.path.exists('imagenes_personalizadas'):
        os.makedirs('imagenes_personalizadas')
    
    print("📤 Sube un archivo ZIP con tus imágenes:")
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall('imagenes_personalizadas')
            print(f"✅ Extraído: {filename}")
    
    imagenes = glob.glob('imagenes_personalizadas/**/*.jpg', recursive=True)
    imagenes.extend(glob.glob('imagenes_personalizadas/**/*.jpeg', recursive=True))
    imagenes.extend(glob.glob('imagenes_personalizadas/**/*.png', recursive=True))
    
    print(f"📊 Encontradas: {len(imagenes)} imágenes")
    return imagenes

def predecir_imagen(model, ruta_imagen):
    """Predice clase de una imagen"""
    img = load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_confidences = predictions[0][top_5_indices]
    
    return predicted_class, confidence, top_5_indices, top_5_confidences

def evaluar_imagenes_personalizadas(model, imagenes):
    """Evalúa imágenes personalizadas"""
    print(f"\n🔍 Evaluando {len(imagenes)} imágenes...")
    
    resultados = []
    for i, ruta_imagen in enumerate(imagenes):
        try:
            clase_predicha, confianza, top5_indices, top5_confianzas = predecir_imagen(model, ruta_imagen)
            
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
    """Función principal"""
    print("=" * 60)
    print("🚗 CLASIFICADOR DE AUTOS CNN - CARS196")
    print("📚 Google Colab Version")
    print("=" * 60)
    
    # 1. Cargar dataset
    train_dataset, test_dataset, info = cargar_dataset()
    
    # 2. Preprocesar datos
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
    
    # 3. Crear modelo
    model, base_model = crear_modelo()
    model.summary()
    
    # 4. Entrenar modelo
    history = entrenar_modelo(model, train_dataset_final, val_dataset, base_model)
    
    # 5. Evaluar modelo
    test_accuracy, test_top5_accuracy = evaluar_modelo(model, test_dataset)
    
    # 6. Visualizar resultados
    visualizar_resultados(history)
    
    # 7. Guardar modelo
    model.save('modelo_autos_final.h5')
    print("\n💾 Modelo guardado: modelo_autos_final.h5")
    
    # 8. Resumen final
    print(f"\n" + "=" * 60)
    print(f"🎉 ENTRENAMIENTO COMPLETADO")
    print(f"=" * 60)
    print(f"📊 Precisión: {test_accuracy:.4f}")
    print(f"🏆 Top-5: {test_top5_accuracy:.4f}")
    
    # 9. Imágenes personalizadas
    print(f"\n" + "=" * 60)
    print(f"🔍 EVALUACIÓN DE IMÁGENES PERSONALIZADAS")
    print(f"=" * 60)
    
    evaluar_personalizadas = input("¿Evaluar imágenes personalizadas? (s/n): ").lower().strip()
    
    if evaluar_personalizadas == 's':
        try:
            imagenes = subir_imagenes_personalizadas()
            if imagenes:
                resultados = evaluar_imagenes_personalizadas(model, imagenes)
                
                # DataFrame con resultados
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
                
                print(f"\n📊 Resumen:")
                print(df_resultados.head(10))
                
                df_resultados.to_csv('resultados_imagenes_personalizadas.csv', index=False)
                print(f"\n💾 Resultados guardados en CSV")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 ¡Proyecto completado!")

# =============================================================================
# EJECUCIÓN
# =============================================================================
if __name__ == "__main__":
    main()
