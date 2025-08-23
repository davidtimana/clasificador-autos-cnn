# -*- coding: utf-8 -*-
"""
Clasificador de Autos CNN - Dataset Cars196
Basado en el curso de Visión por Computador

Este script implementa una Red Neuronal Convolucional (CNN) completa para clasificar
automóviles utilizando el dataset Cars196. Incluye transfer learning, data augmentation,
y técnicas avanzadas de optimización para lograr la mejor precisión posible.

Características principales:
- Transfer Learning con ResNet50V2 pre-entrenado en ImageNet
- Data augmentation para mejorar la generalización
- Fine-tuning automático de las capas del modelo base
- Callbacks avanzados para optimización del entrenamiento
- Evaluación completa con múltiples métricas

Autor: David Timana
Fecha: 2024
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# =============================================================================
# CONFIGURACIÓN DE HARDWARE Y GPU
# =============================================================================
# Verificar si hay GPU disponible para acelerar el entrenamiento
print("GPU disponible:", tf.config.list_physical_devices('GPU'))
if tf.config.list_physical_devices('GPU'):
    # Configurar crecimiento de memoria para evitar errores de memoria GPU
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    print("✅ GPU configurada correctamente")
else:
    print("⚠️  No se detectó GPU, usando CPU (entrenamiento será más lento)")

# =============================================================================
# HIPERPARÁMETROS Y CONFIGURACIÓN
# =============================================================================
# Parámetros de imagen y preprocesamiento
IMG_SIZE = 224  # Tamaño estándar para transfer learning (ResNet, VGG, etc.)
BATCH_SIZE = 32  # Balance entre memoria GPU y estabilidad del entrenamiento
EPOCHS = 50  # Número máximo de épocas (early stopping puede detener antes)
LEARNING_RATE = 0.001  # Learning rate inicial para el optimizador Adam
NUM_CLASSES = 196  # Cars196 tiene exactamente 196 clases de automóviles

# Parámetros de división de datos
TRAIN_SPLIT = 0.8  # 80% de datos para entrenamiento
VALIDATION_SPLIT = 0.2  # 20% de datos para validación

# Parámetros de regularización
DROPOUT_RATE_CONV = 0.25  # Dropout para capas convolucionales
DROPOUT_RATE_DENSE = 0.5  # Dropout para capas densas (más agresivo)

def cargar_dataset_cars196():
    """
    Carga el dataset Cars196 de TensorFlow Datasets
    
    Esta función descarga automáticamente el dataset Cars196 desde TensorFlow Datasets.
    El dataset contiene 16,185 imágenes de 196 clases diferentes de automóviles,
    divididas en conjuntos de entrenamiento y prueba.
    
    Returns:
        train_dataset: Dataset de TensorFlow para entrenamiento
        test_dataset: Dataset de TensorFlow para prueba
        info: Información del dataset (número de ejemplos, clases, etc.)
    """
    print("🔄 Cargando dataset Cars196 desde TensorFlow Datasets...")
    
    # Cargar dataset con información detallada
    # as_supervised=True: Devuelve tuplas (imagen, etiqueta) en lugar de diccionarios
    dataset, info = tfds.load('cars196', 
                             with_info=True,  # Obtener información del dataset
                             as_supervised=True,  # Formato (imagen, etiqueta)
                             split=['train', 'test'])  # Cargar ambos splits
    
    train_dataset, test_dataset = dataset[0], dataset[1]
    
    # Mostrar información detallada del dataset
    print(f"✅ Dataset Cars196 cargado exitosamente:")
    print(f"   📊 Entrenamiento: {info.splits['train'].num_examples:,} imágenes")
    print(f"   📊 Prueba: {info.splits['test'].num_examples:,} imágenes")
    print(f"   🏷️  Clases: {info.features['label'].num_classes} tipos de automóviles")
    print(f"   📁 Tamaño total: ~1.82 GB")
    
    return train_dataset, test_dataset, info

def preprocesar_imagen(image, label):
    """
    Preprocesa las imágenes para el entrenamiento con data augmentation
    
    Esta función aplica transformaciones a las imágenes para mejorar la generalización
    del modelo. Incluye redimensionamiento, normalización y técnicas de data augmentation
    que ayudan a que el modelo sea más robusto a variaciones en las imágenes.
    
    Args:
        image: Imagen de entrada (tensor de TensorFlow)
        label: Etiqueta de la imagen
        
    Returns:
        image: Imagen preprocesada y aumentada
        label: Etiqueta sin cambios
    """
    # =============================================================================
    # REDIMENSIONAMIENTO DE IMAGEN
    # =============================================================================
    # Redimensionar todas las imágenes al tamaño estándar para transfer learning
    # Esto es necesario porque las imágenes originales pueden tener diferentes tamaños
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # =============================================================================
    # NORMALIZACIÓN DE PÍXELES
    # =============================================================================
    # Convertir a float32 y normalizar valores de píxeles al rango [0, 1]
    # Esto es crucial para el entrenamiento estable de redes neuronales
    image = tf.cast(image, tf.float32) / 255.0
    
    # =============================================================================
    # DATA AUGMENTATION (AUMENTACIÓN DE DATOS)
    # =============================================================================
    # Estas técnicas crean variaciones de las imágenes para mejorar la generalización
    
    # 1. FLIP HORIZONTAL (Volteo horizontal)
    # Probabilidad 50% de voltear la imagen horizontalmente
    # Útil porque los autos pueden aparecer desde diferentes ángulos
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_flip_left_right(image)
    
    # 2. AJUSTE DE BRILLO
    # Probabilidad 50% de ajustar el brillo en ±20%
    # Ayuda con diferentes condiciones de iluminación
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, 0.2)
    
    # 3. AJUSTE DE CONTRASTE
    # Probabilidad 50% de ajustar el contraste entre 80% y 120%
    # Mejora la robustez a diferentes condiciones de iluminación
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    
    return image, label

def preprocesar_imagen_test(image, label):
    """
    Preprocesa las imágenes para validación/prueba (sin data augmentation)
    
    Esta función aplica solo las transformaciones básicas necesarias para la evaluación:
    redimensionamiento y normalización. NO incluye data augmentation porque queremos
    evaluar el modelo con las imágenes originales para obtener métricas reales.
    
    Args:
        image: Imagen de entrada (tensor de TensorFlow)
        label: Etiqueta de la imagen
        
    Returns:
        image: Imagen preprocesada (sin augmentation)
        label: Etiqueta sin cambios
    """
    # =============================================================================
    # REDIMENSIONAMIENTO DE IMAGEN
    # =============================================================================
    # Redimensionar al mismo tamaño que las imágenes de entrenamiento
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # =============================================================================
    # NORMALIZACIÓN DE PÍXELES
    # =============================================================================
    # Aplicar la misma normalización que en entrenamiento
    # Esto es crucial para mantener consistencia entre entrenamiento y evaluación
    image = tf.cast(image, tf.float32) / 255.0
    
    # NOTA: NO aplicamos data augmentation aquí porque:
    # 1. Queremos evaluar con imágenes originales
    # 2. Las métricas deben ser consistentes y reproducibles
    # 3. La evaluación debe reflejar el rendimiento real del modelo
    
    return image, label

def crear_arquitectura_cnn():
    """
    Crea una arquitectura CNN personalizada optimizada para clasificación de autos
    
    Esta función construye una red neuronal convolucional desde cero, diseñada
    específicamente para el dataset Cars196. La arquitectura incluye técnicas
    modernas como Batch Normalization y Dropout para mejorar el entrenamiento
    y prevenir overfitting.
    
    Returns:
        model: Modelo de Keras compilado y listo para entrenar
    """
    print("🏗️  Creando arquitectura CNN personalizada...")
    
    # =============================================================================
    # ARQUITECTURA CNN PERSONALIZADA
    # =============================================================================
    # Esta arquitectura está diseñada específicamente para clasificación de autos
    # con un balance entre capacidad de aprendizaje y eficiencia computacional
    
    model = models.Sequential([
        # =============================================================================
        # PRIMERA CAPA CONVOLUCIONAL
        # =============================================================================
        # Conv2D(32, (3,3)): 32 filtros de 3x3 píxeles
        # - 32 filtros: Detecta características básicas (bordes, texturas)
        # - (3,3): Tamaño del kernel (receptivo field pequeño)
        # - activation='relu': Función de activación no lineal
        # - input_shape: Especifica el tamaño de entrada (224x224x3)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # BatchNormalization: Normaliza las activaciones para estabilizar el entrenamiento
        # - Reduce el problema de covariate shift
        # - Permite usar learning rates más altos
        layers.BatchNormalization(),
        
        # MaxPooling2D: Reduce la dimensionalidad espacial
        # - (2,2): Toma el máximo valor en ventanas de 2x2
        # - Reduce el tamaño de la imagen a la mitad
        layers.MaxPooling2D((2, 2)),
        
        # Dropout: Previene overfitting desactivando aleatoriamente neuronas
        # - 25% de las neuronas se desactivan durante el entrenamiento
        layers.Dropout(DROPOUT_RATE_CONV),
        
        # =============================================================================
        # SEGUNDA CAPA CONVOLUCIONAL
        # =============================================================================
        # 64 filtros: Detecta características más complejas
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE_CONV),
        
        # =============================================================================
        # TERCERA CAPA CONVOLUCIONAL
        # =============================================================================
        # 128 filtros: Detecta características de nivel medio
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE_CONV),
        
        # =============================================================================
        # CUARTA CAPA CONVOLUCIONAL
        # =============================================================================
        # 256 filtros: Detecta características de alto nivel
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(DROPOUT_RATE_CONV),
        
        # =============================================================================
        # CAPAS DENSAS (FULLY CONNECTED)
        # =============================================================================
        # Flatten: Convierte el tensor 2D en un vector 1D
        # Necesario para conectar con capas densas
        layers.Flatten(),
        
        # Primera capa densa: 512 neuronas
        # - Procesa las características extraídas por las capas convolucionales
        # - 512 es un buen balance entre capacidad y eficiencia
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE_DENSE),  # Dropout más agresivo en capas densas
        
        # Segunda capa densa: 256 neuronas
        # - Reduce la dimensionalidad gradualmente
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE_DENSE),
        
        # Capa de salida: NUM_CLASSES neuronas (196 para Cars196)
        # - activation='softmax': Convierte salidas en probabilidades
        # - Las probabilidades suman 1.0
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print("✅ Arquitectura CNN personalizada creada exitosamente")
    return model

def crear_modelo_transfer_learning():
    """
    Crea un modelo usando transfer learning con ResNet50V2
    
    Esta función implementa transfer learning, una técnica poderosa que aprovecha
    el conocimiento aprendido por modelos pre-entrenados en datasets grandes (como ImageNet)
    y lo adapta para nuestro problema específico de clasificación de autos.
    
    Ventajas del transfer learning:
    - Entrenamiento más rápido
    - Mejor generalización
    - Requiere menos datos
    - Mejores resultados finales
    
    Returns:
        model: Modelo completo con ResNet50V2 + capas de clasificación
        base_model: Modelo base ResNet50V2 (para fine-tuning posterior)
    """
    print("🚀 Creando modelo con Transfer Learning (ResNet50V2)...")
    
    # =============================================================================
    # CARGA DEL MODELO BASE PRE-ENTRENADO
    # =============================================================================
    # ResNet50V2: Arquitectura moderna y eficiente
    # - weights='imagenet': Usar pesos pre-entrenados en ImageNet (1.2M imágenes)
    # - include_top=False: No incluir las capas de clasificación finales
    # - input_shape: Especificar el tamaño de entrada (224x224x3)
    base_model = ResNet50V2(
        weights='imagenet',  # Pesos pre-entrenados en ImageNet
        include_top=False,   # Sin capas de clasificación finales
        input_shape=(IMG_SIZE, IMG_SIZE, 3)  # Tamaño de entrada
    )
    
    # =============================================================================
    # CONGELAR CAPAS DEL MODELO BASE
    # =============================================================================
    # Durante la primera fase del entrenamiento, mantenemos los pesos del modelo base
    # congelados para aprovechar el conocimiento pre-entrenado
    base_model.trainable = False
    print(f"   🔒 Capas del modelo base congeladas: {len(base_model.layers)} capas")
    
    # =============================================================================
    # CONSTRUCCIÓN DEL MODELO COMPLETO
    # =============================================================================
    model = models.Sequential([
        # =============================================================================
        # MODELO BASE (ResNet50V2)
        # =============================================================================
        # Este modelo ya sabe extraer características generales de imágenes
        # (bordes, texturas, formas, etc.) gracias al entrenamiento en ImageNet
        base_model,
        
        # =============================================================================
        # GLOBAL AVERAGE POOLING
        # =============================================================================
        # Convierte el tensor de características en un vector
        # - Reduce significativamente el número de parámetros
        # - Mantiene información espacial importante
        # - Más eficiente que Flatten() para características convolucionales
        layers.GlobalAveragePooling2D(),
        
        # =============================================================================
        # BATCH NORMALIZATION
        # =============================================================================
        # Normaliza las activaciones para estabilizar el entrenamiento
        layers.BatchNormalization(),
        
        # =============================================================================
        # DROPOUT PARA REGULARIZACIÓN
        # =============================================================================
        # Previene overfitting desactivando 50% de las neuronas
        layers.Dropout(DROPOUT_RATE_DENSE),
        
        # =============================================================================
        # PRIMERA CAPA DENSA
        # =============================================================================
        # 512 neuronas: Procesa las características extraídas por ResNet50V2
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE_DENSE),
        
        # =============================================================================
        # SEGUNDA CAPA DENSA
        # =============================================================================
        # 256 neuronas: Reduce dimensionalidad gradualmente
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE_DENSE),
        
        # =============================================================================
        # CAPA DE SALIDA
        # =============================================================================
        # NUM_CLASSES neuronas (196 para Cars196)
        # Softmax convierte las salidas en probabilidades
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    print("✅ Modelo de transfer learning creado exitosamente")
    print(f"   📊 Parámetros totales: {model.count_params():,}")
    print(f"   🔒 Parámetros congelados: {sum([tf.size(w).numpy() for w in base_model.weights]):,}")
    print(f"   🔓 Parámetros entrenables: {model.count_params() - sum([tf.size(w).numpy() for w in base_model.weights]):,}")
    
    return model, base_model

def compilar_y_entrenar_modelo(model, train_dataset, val_dataset, use_transfer_learning=False):
    """
    Compila y entrena el modelo
    """
    print("Compilando modelo...")
    
    # Optimizador
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Compilar modelo
    model.compile(
        optimizer=optimizer,
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
    
    print("Iniciando entrenamiento...")
    
    # Entrenamiento inicial
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Si es transfer learning, hacer fine-tuning
    if use_transfer_learning:
        print("Iniciando fine-tuning...")
        
        # Descongelar algunas capas del modelo base
        base_model = model.layers[0]
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
            epochs=20,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combinar historiales
        for key in history.history:
            history.history[key].extend(history_fine.history[key])
    
    return history

def evaluar_modelo(model, test_dataset, info):
    """
    Evalúa el modelo en el conjunto de prueba
    """
    print("Evaluando modelo...")
    
    # Evaluación
    test_loss, test_accuracy, test_top5_accuracy = model.evaluate(test_dataset, verbose=1)
    
    print(f"\nResultados de evaluación:")
    print(f"Pérdida de prueba: {test_loss:.4f}")
    print(f"Precisión de prueba: {test_accuracy:.4f}")
    print(f"Top-5 precisión: {test_top5_accuracy:.4f}")
    
    # Predicciones para análisis detallado
    predictions = []
    true_labels = []
    
    for images, labels in test_dataset.take(100):  # Evaluar en 100 ejemplos
        pred = model.predict(images, verbose=0)
        predictions.extend(np.argmax(pred, axis=1))
        true_labels.extend(labels.numpy())
    
    # Matriz de confusión
    cm = confusion_matrix(true_labels, predictions)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    return test_accuracy, test_top5_accuracy

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
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()

def predecir_imagen_personalizada(model, ruta_imagen):
    """
    Predice la clase de una imagen personalizada
    """
    # Cargar y preprocesar imagen
    img = tf.keras.preprocessing.image.load_img(ruta_imagen, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    
    # Predicción
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    # Top-5 predicciones
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_confidences = predictions[0][top_5_indices]
    
    return predicted_class, confidence, top_5_indices, top_5_confidences

def main():
    """
    Función principal
    """
    print("=== CLASIFICADOR DE AUTOS CNN - DATASET CARS196 ===")
    print("Basado en el curso de Visión por Computador\n")
    
    # 1. Cargar dataset
    train_dataset, test_dataset, info = cargar_dataset_cars196()
    
    # 2. Preprocesar datos
    print("Preprocesando datos...")
    
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
    
    # 3. Crear modelo
    print("\nSeleccionando arquitectura...")
    print("1. Arquitectura personalizada (más rápida)")
    print("2. Transfer Learning con ResNet50V2 (mejor precisión)")
    
    # Por defecto usar transfer learning para mejor rendimiento
    use_transfer_learning = True
    
    if use_transfer_learning:
        model, base_model = crear_modelo_transfer_learning()
    else:
        model = crear_arquitectura_cnn()
    
    # Mostrar resumen del modelo
    model.summary()
    
    # 4. Entrenar modelo
    history = compilar_y_entrenar_modelo(
        model, 
        train_dataset_final, 
        val_dataset, 
        use_transfer_learning=use_transfer_learning
    )
    
    # 5. Evaluar modelo
    test_accuracy, test_top5_accuracy = evaluar_modelo(model, test_dataset, info)
    
    # 6. Visualizar resultados
    visualizar_resultados(history)
    
    # 7. Guardar modelo
    model.save('modelo_autos_final.h5')
    print("\nModelo guardado como 'modelo_autos_final.h5'")
    
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Precisión final: {test_accuracy:.4f}")
    print(f"Top-5 precisión: {test_top5_accuracy:.4f}")
    print(f"Modelo listo para usar con imágenes personalizadas!")
    
    return model, history

if __name__ == "__main__":
    # Ejecutar entrenamiento
    modelo_entrenado, historial = main()
    
    # Ejemplo de uso con imagen personalizada
    print("\n=== EJEMPLO DE USO CON IMAGEN PERSONALIZADA ===")
    print("Para usar el modelo con tus 100 imágenes personalizadas:")
    print("1. Coloca las imágenes en una carpeta")
    print("2. Usa la función predecir_imagen_personalizada()")
    print("3. Ejemplo:")
    print("   ruta_imagen = 'tu_imagen.jpg'")
    print("   clase_predicha, confianza, top5_indices, top5_confianzas = predecir_imagen_personalizada(modelo_entrenado, ruta_imagen)")
