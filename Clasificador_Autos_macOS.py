# -*- coding: utf-8 -*-
"""
Clasificador de Autos CNN - Versión macOS
Optimizado para evitar problemas de compatibilidad en macOS
Autor: David Timana | Curso: Visión por Computador
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Configuraciones específicas para macOS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Importaciones
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print("🚗 CLASIFICADOR DE AUTOS CNN - VERSIÓN macOS")
print("=" * 50)

# Verificar configuración
print(f"✅ TensorFlow version: {tf.__version__}")
print(f"✅ Python version: {tf.version.VERSION}")
print(f"✅ GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# 1. CARGAR DATASET CIFAR-10
# =============================================================================
print("\n1. Cargando dataset CIFAR-10...")

try:
    # Cargar CIFAR-10 directamente desde Keras
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    print(f"✅ Dataset cargado exitosamente:")
    print(f"   📊 Entrenamiento: {x_train.shape[0]:,} imágenes")
    print(f"   📊 Prueba: {x_test.shape[0]:,} imágenes")
    print(f"   🖼️  Tamaño: {x_train.shape[1]}x{x_train.shape[2]}x{x_train.shape[3]}")
    print(f"   🏷️  Clases: 10")
    
    # Mostrar clases disponibles
    class_names = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']
    print(f"   📝 Clases: {class_names}")
    print(f"   🚗 Automóviles: clase 1")
    
except Exception as e:
    print(f"❌ Error al cargar dataset: {e}")
    exit(1)

# =============================================================================
# 2. PREPROCESAR DATOS
# =============================================================================
print("\n2. Preprocesando datos...")

try:
    # Normalizar imágenes (0-1)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convertir etiquetas a one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"✅ Datos preprocesados:")
    print(f"   📊 x_train shape: {x_train.shape}")
    print(f"   📊 y_train shape: {y_train.shape}")
    print(f"   📊 x_test shape: {x_test.shape}")
    print(f"   📊 y_test shape: {y_test.shape}")
    
except Exception as e:
    print(f"❌ Error en preprocesamiento: {e}")
    exit(1)

# =============================================================================
# 3. CONSTRUIR MODELO CNN
# =============================================================================
print("\n3. Construyendo modelo CNN...")

try:
    model = Sequential([
        # Primera capa convolucional
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Segunda capa convolucional
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Tercera capa convolucional
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Aplanar
        Flatten(),
        
        # Capas densas
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])
    
    print("✅ Modelo creado exitosamente:")
    model.summary()
    
except Exception as e:
    print(f"❌ Error al crear modelo: {e}")
    exit(1)

# =============================================================================
# 4. COMPILAR Y ENTRENAR MODELO
# =============================================================================
print("\n4. Compilando y entrenando modelo...")

try:
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar modelo
    print("🚀 Iniciando entrenamiento...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=1
    )
    
    print("✅ Entrenamiento completado exitosamente!")
    
except Exception as e:
    print(f"❌ Error durante entrenamiento: {e}")
    exit(1)

# =============================================================================
# 5. EVALUAR MODELO
# =============================================================================
print("\n5. Evaluando modelo...")

try:
    # Evaluar en conjunto de prueba
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"📈 Resultados:")
    print(f"   📉 Test loss: {score[0]:.4f}")
    print(f"   ✅ Test accuracy: {score[1]:.4f}")
    
except Exception as e:
    print(f"❌ Error en evaluación: {e}")
    exit(1)

# =============================================================================
# 6. VISUALIZAR RESULTADOS
# =============================================================================
print("\n6. Visualizando resultados...")

try:
    # Gráfico de precisión
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones completadas!")
    
except Exception as e:
    print(f"❌ Error en visualización: {e}")

# =============================================================================
# 7. PREDICCIONES EN IMÁGENES ESPECÍFICAS
# =============================================================================
print("\n7. Predicciones en imágenes específicas...")

try:
    # Seleccionar algunas imágenes de prueba
    n = 5  # Índice de imagen a probar
    im1 = x_test[n:n+1]  # Tomar una imagen
    
    # Mostrar imagen
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[n])
    plt.title(f'Imagen de prueba (Clase real: {np.argmax(y_test[n])})')
    plt.axis('off')
    
    # Hacer predicción
    Yprod = model.predict(im1)
    predicted_class = np.argmax(Yprod[0])
    confidence = np.max(Yprod[0])
    
    print(f"📊 Predicción:")
    print(f"   🖼️  Imagen: {n}")
    print(f"   🎯 Clase predicha: {predicted_class} ({class_names[predicted_class]})")
    print(f"   📈 Confianza: {confidence:.4f}")
    print(f"   ✅ Clase real: {np.argmax(y_test[n])} ({class_names[np.argmax(y_test[n])]})")
    
    # Mostrar probabilidades
    plt.subplot(1, 2, 2)
    plt.bar(range(10), Yprod[0])
    plt.title('Probabilidades por Clase')
    plt.xlabel('Clase')
    plt.ylabel('Probabilidad')
    plt.xticks(range(10), [f'{i}\n{name[:3]}' for i, name in enumerate(class_names)], fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Predicciones completadas!")
    
except Exception as e:
    print(f"❌ Error en predicciones: {e}")

# =============================================================================
# 8. ANÁLISIS DE CLASE ESPECÍFICA (AUTOMÓVILES)
# =============================================================================
print("\n8. Análisis específico de automóviles...")

try:
    # Filtrar solo automóviles (clase 1)
    car_indices = np.where(np.argmax(y_test, axis=1) == 1)[0]
    car_images = x_test[car_indices]
    car_labels = y_test[car_indices]
    
    print(f"🚗 Automóviles en conjunto de prueba: {len(car_indices)}")
    
    # Evaluar precisión específica en automóviles
    car_predictions = model.predict(car_images)
    car_accuracy = np.mean(np.argmax(car_predictions, axis=1) == 1)
    print(f"✅ Precisión en automóviles: {car_accuracy:.4f}")
    
    # Mostrar algunos automóviles y sus predicciones
    plt.figure(figsize=(15, 5))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(car_images[i])
        pred_class = np.argmax(car_predictions[i])
        pred_conf = np.max(car_predictions[i])
        plt.title(f'Pred: {pred_class}\nConf: {pred_conf:.2f}')
        plt.axis('off')
    
    plt.suptitle('Predicciones en Automóviles', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print("✅ Análisis de automóviles completado!")
    
except Exception as e:
    print(f"❌ Error en análisis de automóviles: {e}")

# =============================================================================
# 9. MATRIZ DE CONFUSIÓN SIMPLIFICADA
# =============================================================================
print("\n9. Matriz de confusión...")

try:
    # Predicciones en todo el conjunto de prueba
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Contar aciertos por clase
    print("📊 Aciertos por clase:")
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true_classes == i)[0]
        if len(class_indices) > 0:
            class_accuracy = np.mean(y_pred_classes[class_indices] == i)
            print(f"   {i:2d}. {class_name:10s}: {class_accuracy:.4f}")
    
    print("✅ Matriz de confusión completada!")
    
except Exception as e:
    print(f"❌ Error en matriz de confusión: {e}")

# =============================================================================
# 10. GUARDAR MODELO
# =============================================================================
print("\n10. Guardando modelo...")

try:
    model.save('modelo_autos_cnn_macos.h5')
    print("✅ Modelo guardado como 'modelo_autos_cnn_macos.h5'")
    
except Exception as e:
    print(f"❌ Error al guardar modelo: {e}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 50)
print("🎉 ENTRENAMIENTO COMPLETADO")
print("=" * 50)
print(f"📊 Precisión final: {score[1]:.4f}")
print(f"📉 Pérdida final: {score[0]:.4f}")
print(f"🏷️  Clases: 10 (incluyendo automóviles)")
print(f"🚗 Precisión en automóviles: {car_accuracy:.4f}")
print(f"💾 Modelo guardado: modelo_autos_cnn_macos.h5")

print("\n🎯 El modelo puede clasificar:")
for i, name in enumerate(class_names):
    print(f"   {i}. {name}")

print("\n✅ ¡Proyecto completado exitosamente en macOS!")
