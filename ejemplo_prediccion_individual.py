# -*- coding: utf-8 -*-
"""
Ejemplo de Predicción Individual - Clasificador de Autos
Script de ejemplo para hacer predicciones en imágenes individuales

Este script proporciona una interfaz simple y directa para hacer predicciones
en imágenes individuales usando el modelo CNN entrenado en el dataset Cars196.
Es ideal para pruebas rápidas y demostraciones del clasificador.

Características:
- Predicción rápida de imágenes individuales
- Visualización de imagen + Top-5 predicciones
- Modo interactivo para probar múltiples imágenes
- Interfaz de usuario amigable
- Manejo de errores robusto

Autor: David Timana
Fecha: 2024
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# =============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS
# =============================================================================
# Parámetros de imagen (deben coincidir con el entrenamiento)
IMG_SIZE = 224  # Tamaño de imagen usado en el entrenamiento

# Configuración de visualización
plt.style.use('seaborn-v0_8')  # Estilo moderno para los gráficos

def cargar_y_preprocesar_imagen(ruta_imagen, img_size=224):
    """
    Carga y preprocesa una imagen individual para predicción
    
    Esta función toma una imagen desde el sistema de archivos, la carga,
    redimensiona y normaliza para que sea compatible con el modelo CNN
    entrenado. Es el primer paso necesario antes de hacer cualquier predicción.
    
    Args:
        ruta_imagen (str): Ruta completa al archivo de imagen
        img_size (int): Tamaño al que redimensionar la imagen (debe coincidir con el entrenamiento)
        
    Returns:
        img_array (numpy.ndarray): Array de la imagen preprocesada lista para predicción
        img (PIL.Image): Imagen original cargada (para visualización)
    """
    # =============================================================================
    # CARGA DE IMAGEN
    # =============================================================================
    # Cargar imagen usando Keras (más eficiente para este caso)
    # target_size: Redimensionar automáticamente al tamaño requerido por el modelo
    img = tf.keras.preprocessing.image.load_img(ruta_imagen, target_size=(img_size, img_size))
    
    # =============================================================================
    # CONVERSIÓN A ARRAY Y NORMALIZACIÓN
    # =============================================================================
    # Convertir imagen PIL a array numpy
    # El array tendrá forma (img_size, img_size, 3) para imágenes RGB
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    
    # Normalizar valores de píxeles al rango [0, 1]
    # Esto es crucial para mantener consistencia con el entrenamiento
    # Los valores originales están en [0, 255], los dividimos por 255
    img_array = img_array / 255.0
    
    # =============================================================================
    # PREPARACIÓN PARA PREDICCIÓN
    # =============================================================================
    # Agregar dimensión de batch (batch_size=1)
    # El modelo espera entrada de forma (batch_size, height, width, channels)
    # expand_dims agrega una dimensión al inicio: (1, img_size, img_size, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array, img

def predecir_auto(modelo, ruta_imagen):
    """
    Predice la clase de un automóvil en una imagen
    
    Esta es la función principal que realiza la predicción. Toma una imagen,
    la preprocesa y usa el modelo CNN entrenado para predecir la clase del
    automóvil. Devuelve tanto la predicción principal como las Top-5 predicciones
    con sus respectivas confianzas.
    
    Args:
        modelo: Modelo CNN entrenado (Keras model)
        ruta_imagen (str): Ruta al archivo de imagen
        
    Returns:
        dict: Diccionario con todos los resultados de la predicción:
            - imagen_original: Imagen cargada (para visualización)
            - clase_predicha: Índice de la clase predicha (0-195)
            - confianza: Confianza de la predicción principal (0-1)
            - top5_clases: Array con los índices de las 5 mejores predicciones
            - top5_confianzas: Array con las confianzas de las 5 mejores predicciones
            - todas_predicciones: Array completo con todas las predicciones (196 valores)
    """
    # =============================================================================
    # PREPROCESAMIENTO DE LA IMAGEN
    # =============================================================================
    # Cargar y preprocesar la imagen para que sea compatible con el modelo
    img_array, img_original = cargar_y_preprocesar_imagen(ruta_imagen)
    
    # =============================================================================
    # PREDICCIÓN CON EL MODELO
    # =============================================================================
    # Realizar predicción usando el modelo CNN
    # verbose=0: No mostrar barra de progreso (predicción individual)
    predicciones = modelo.predict(img_array, verbose=0)
    
    # =============================================================================
    # PROCESAMIENTO DE RESULTADOS
    # =============================================================================
    # Obtener la clase predicha (índice con mayor probabilidad)
    clase_predicha = np.argmax(predicciones[0])
    
    # Obtener la confianza de la predicción principal
    confianza = np.max(predicciones[0])
    
    # =============================================================================
    # TOP-5 PREDICCIONES
    # =============================================================================
    # Obtener los índices de las 5 clases con mayor probabilidad
    # argsort ordena de menor a mayor, tomamos los últimos 5 y los invertimos
    top5_indices = np.argsort(predicciones[0])[-5:][::-1]
    
    # Obtener las confianzas correspondientes a las Top-5 predicciones
    top5_confianzas = predicciones[0][top5_indices]
    
    # =============================================================================
    # PREPARACIÓN DEL RESULTADO
    # =============================================================================
    # Crear diccionario con todos los resultados para facilitar el uso
    resultado = {
        'imagen_original': img_original,      # Para visualización
        'clase_predicha': clase_predicha,     # Clase principal predicha
        'confianza': confianza,               # Confianza de la predicción principal
        'top5_clases': top5_indices,          # Top-5 clases predichas
        'top5_confianzas': top5_confianzas,   # Confianzas de Top-5
        'todas_predicciones': predicciones[0] # Todas las predicciones (196 valores)
    }
    
    return resultado

def visualizar_prediccion(resultado, ruta_imagen):
    """
    Visualiza la imagen y las predicciones
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Mostrar imagen original
    ax1.imshow(resultado['imagen_original'])
    ax1.set_title(f'Imagen: {os.path.basename(ruta_imagen)}')
    ax1.axis('off')
    
    # Mostrar top-5 predicciones
    clases = resultado['top5_clases']
    confianzas = resultado['top5_confianzas']
    
    y_pos = np.arange(len(clases))
    ax2.barh(y_pos, confianzas)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'Clase {clase}' for clase in clases])
    ax2.set_xlabel('Confianza')
    ax2.set_title('Top-5 Predicciones')
    ax2.invert_yaxis()
    
    # Agregar valores de confianza en las barras
    for i, v in enumerate(confianzas):
        ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # Imprimir resultados
    print(f"\n=== PREDICCIÓN PARA: {os.path.basename(ruta_imagen)} ===")
    print(f"Clase predicha: {resultado['clase_predicha']}")
    print(f"Confianza: {resultado['confianza']:.4f}")
    print(f"\nTop-5 predicciones:")
    for i, (clase, conf) in enumerate(zip(clases, confianzas)):
        print(f"{i+1}. Clase {clase}: {conf:.4f}")

def ejemplo_uso():
    """
    Ejemplo de uso del clasificador
    """
    print("=== EJEMPLO DE PREDICCIÓN INDIVIDUAL ===")
    print("Clasificador de Autos - Dataset Cars196\n")
    
    # 1. Cargar modelo entrenado
    try:
        modelo = load_model('modelo_autos_final.h5')
        print("✅ Modelo cargado exitosamente")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo 'modelo_autos_final.h5'")
        print("   Primero ejecuta: python clasificador_autos_cnn.py")
        return
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # 2. Solicitar ruta de imagen
    ruta_imagen = input("Ingresa la ruta de la imagen a clasificar: ").strip()
    
    if not os.path.exists(ruta_imagen):
        print(f"❌ Error: La imagen '{ruta_imagen}' no existe")
        return
    
    # 3. Hacer predicción
    try:
        resultado = predecir_auto(modelo, ruta_imagen)
        print("✅ Predicción realizada exitosamente")
    except Exception as e:
        print(f"❌ Error durante la predicción: {e}")
        return
    
    # 4. Visualizar resultados
    visualizar_prediccion(resultado, ruta_imagen)
    
    # 5. Información adicional
    print(f"\n=== INFORMACIÓN ADICIONAL ===")
    print(f"El modelo puede clasificar 196 diferentes tipos de automóviles")
    print(f"Las clases incluyen diferentes marcas, modelos y años")
    print(f"Para más detalles sobre las clases, consulta la documentación del dataset Cars196")

def probar_multiples_imagenes():
    """
    Función para probar múltiples imágenes de una carpeta
    """
    print("=== PRUEBA DE MÚLTIPLES IMÁGENES ===")
    
    # Cargar modelo
    try:
        modelo = load_model('modelo_autos_final.h5')
    except:
        print("❌ Error: No se pudo cargar el modelo")
        return
    
    # Solicitar carpeta
    carpeta = input("Ingresa la ruta de la carpeta con imágenes: ").strip()
    
    if not os.path.exists(carpeta):
        print(f"❌ Error: La carpeta '{carpeta}' no existe")
        return
    
    # Buscar imágenes
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp']
    imagenes = []
    
    for archivo in os.listdir(carpeta):
        if any(archivo.lower().endswith(ext) for ext in extensiones):
            imagenes.append(os.path.join(carpeta, archivo))
    
    if not imagenes:
        print("❌ No se encontraron imágenes en la carpeta")
        return
    
    print(f"Encontradas {len(imagenes)} imágenes")
    
    # Procesar cada imagen
    for i, ruta_imagen in enumerate(imagenes[:5]):  # Solo las primeras 5
        print(f"\n--- Imagen {i+1}: {os.path.basename(ruta_imagen)} ---")
        
        try:
            resultado = predecir_auto(modelo, ruta_imagen)
            print(f"Clase predicha: {resultado['clase_predicha']}")
            print(f"Confianza: {resultado['confianza']:.4f}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    print("Selecciona una opción:")
    print("1. Predicción individual")
    print("2. Probar múltiples imágenes")
    
    opcion = input("Opción (1 o 2): ").strip()
    
    if opcion == "1":
        ejemplo_uso()
    elif opcion == "2":
        probar_multiples_imagenes()
    else:
        print("Opción no válida")
