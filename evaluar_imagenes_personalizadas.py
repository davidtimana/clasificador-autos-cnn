# -*- coding: utf-8 -*-
"""
Evaluador de Imágenes Personalizadas - Clasificador de Autos
Script para evaluar las 100 imágenes personalizadas con el modelo entrenado

Este script está diseñado específicamente para evaluar un conjunto de imágenes
personalizadas (como las 100 imágenes mencionadas) con el modelo CNN entrenado
en el dataset Cars196. Proporciona análisis detallados, visualizaciones y
reportes completos de las predicciones.

Características principales:
- Carga automática de múltiples formatos de imagen
- Predicciones en lote con Top-5 resultados
- Generación de reportes detallados en CSV
- Visualización de predicciones y estadísticas
- Análisis de distribución de clases y confianzas
- Evaluación con etiquetas reales (opcional)

Autor: David Timana
Fecha: 2024
"""

# =============================================================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# =============================================================================
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
import cv2
from PIL import Image
import glob

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS
# =============================================================================
# Parámetros de imagen (deben coincidir con el entrenamiento)
IMG_SIZE = 224  # Tamaño de imagen usado en el entrenamiento
BATCH_SIZE = 32  # Tamaño de lote para predicciones

# Configuración de visualización
PLOT_STYLE = 'seaborn-v0_8'  # Estilo de gráficos
plt.style.use(PLOT_STYLE)

def cargar_modelo_entrenado(ruta_modelo='modelo_autos_final.h5'):
    """
    Carga el modelo CNN entrenado desde un archivo .h5
    
    Esta función carga el modelo guardado después del entrenamiento. El modelo
    debe haber sido entrenado con la misma arquitectura y parámetros que se
    usan en este script de evaluación.
    
    Args:
        ruta_modelo (str): Ruta al archivo del modelo (.h5)
        
    Returns:
        modelo: Modelo de Keras cargado y listo para predicciones
        None: Si hay error al cargar el modelo
    """
    try:
        print(f"🔄 Cargando modelo desde: {ruta_modelo}")
        
        # Cargar el modelo usando Keras
        modelo = load_model(ruta_modelo)
        
        # Verificar que el modelo se cargó correctamente
        print(f"✅ Modelo cargado exitosamente")
        print(f"   📊 Arquitectura: {modelo.name}")
        print(f"   🔢 Parámetros: {modelo.count_params():,}")
        print(f"   🏷️  Clases de salida: {modelo.output_shape[-1]}")
        
        return modelo
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ruta_modelo}'")
        print("   💡 Asegúrate de haber ejecutado primero: python clasificador_autos_cnn.py")
        return None
        
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        print("   💡 Verifica que el archivo del modelo no esté corrupto")
        return None

def cargar_imagenes_personalizadas(carpeta_imagenes):
    """
    Carga las imágenes personalizadas desde una carpeta específica
    
    Esta función busca automáticamente imágenes en diferentes formatos dentro
    de la carpeta especificada, las carga, preprocesa y prepara para la evaluación
    con el modelo CNN entrenado.
    
    Formatos soportados: JPG, JPEG, PNG, BMP, TIFF (mayúsculas y minúsculas)
    
    Args:
        carpeta_imagenes (str): Ruta a la carpeta que contiene las imágenes
        
    Returns:
        imagenes (list): Lista de arrays numpy con las imágenes preprocesadas
        rutas_imagenes (list): Lista de rutas de las imágenes cargadas exitosamente
    """
    print(f"🔄 Cargando imágenes desde: {carpeta_imagenes}")
    
    # =============================================================================
    # BÚSQUEDA DE ARCHIVOS DE IMAGEN
    # =============================================================================
    # Extensiones de imagen soportadas (formato glob)
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    
    imagenes = []  # Lista para almacenar las imágenes preprocesadas
    rutas_imagenes = []  # Lista para almacenar las rutas de las imágenes
    
    # Buscar imágenes con diferentes extensiones (mayúsculas y minúsculas)
    for extension in extensiones:
        # Buscar archivos con extensión en minúsculas
        rutas = glob.glob(os.path.join(carpeta_imagenes, extension))
        # Buscar archivos con extensión en mayúsculas
        rutas.extend(glob.glob(os.path.join(carpeta_imagenes, extension.upper())))
        rutas_imagenes.extend(rutas)
    
    # Verificar si se encontraron imágenes
    if not rutas_imagenes:
        print("❌ No se encontraron imágenes en la carpeta especificada")
        print("   💡 Verifica que la carpeta contenga archivos con extensiones: .jpg, .jpeg, .png, .bmp, .tiff")
        return [], []
    
    print(f"✅ Encontradas {len(rutas_imagenes)} imágenes")
    
    # =============================================================================
    # CARGA Y PREPROCESAMIENTO DE IMÁGENES
    # =============================================================================
    imagenes_cargadas = 0
    imagenes_fallidas = 0
    
    for ruta in rutas_imagenes:
        try:
            # =============================================================================
            # CARGA DE IMAGEN
            # =============================================================================
            # Cargar imagen usando Keras (más eficiente que PIL para este caso)
            # target_size: Redimensionar automáticamente al tamaño requerido
            img = tf.keras.preprocessing.image.load_img(ruta, target_size=(IMG_SIZE, IMG_SIZE))
            
            # =============================================================================
            # CONVERSIÓN A ARRAY Y NORMALIZACIÓN
            # =============================================================================
            # Convertir imagen PIL a array numpy
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Normalizar valores de píxeles al rango [0, 1]
            # Esto es crucial para mantener consistencia con el entrenamiento
            img_array = img_array / 255.0
            
            # Agregar a las listas
            imagenes.append(img_array)
            imagenes_cargadas += 1
            
        except Exception as e:
            print(f"⚠️  Error al cargar imagen {os.path.basename(ruta)}: {e}")
            imagenes_fallidas += 1
    
    # =============================================================================
    # RESUMEN DE CARGA
    # =============================================================================
    print(f"📊 Resumen de carga:")
    print(f"   ✅ Imágenes cargadas exitosamente: {imagenes_cargadas}")
    print(f"   ❌ Imágenes con errores: {imagenes_fallidas}")
    print(f"   📁 Total de archivos encontrados: {len(rutas_imagenes)}")
    
    if imagenes_cargadas == 0:
        print("❌ No se pudieron cargar ninguna imagen")
        return [], []
    
    return imagenes, rutas_imagenes

def predecir_lote_imagenes(modelo, imagenes, rutas_imagenes):
    """
    Predice las clases para un lote de imágenes
    """
    if not imagenes:
        return [], []
    
    # Convertir a array numpy
    imagenes_array = np.array(imagenes)
    
    # Predicciones
    predicciones = modelo.predict(imagenes_array, verbose=1)
    
    # Obtener clases predichas y confianzas
    clases_predichas = np.argmax(predicciones, axis=1)
    confianzas = np.max(predicciones, axis=1)
    
    # Top-5 predicciones para cada imagen
    top5_indices = np.argsort(predicciones, axis=1)[:, -5:][:, ::-1]
    top5_confianzas = np.take_along_axis(predicciones, top5_indices, axis=1)
    
    return clases_predichas, confianzas, top5_indices, top5_confianzas

def crear_reporte_detallado(rutas_imagenes, clases_predichas, confianzas, top5_indices, top5_confianzas):
    """
    Crea un reporte detallado de las predicciones
    """
    # Crear DataFrame con resultados
    resultados = []
    
    for i, ruta in enumerate(rutas_imagenes):
        nombre_archivo = os.path.basename(ruta)
        
        # Top-5 predicciones
        top5_pred = top5_indices[i]
        top5_conf = top5_confianzas[i]
        
        resultado = {
            'Archivo': nombre_archivo,
            'Ruta': ruta,
            'Clase_Predicha': int(clases_predichas[i]),
            'Confianza': float(confianzas[i]),
            'Top1_Clase': int(top5_pred[0]),
            'Top1_Confianza': float(top5_conf[0]),
            'Top2_Clase': int(top5_pred[1]),
            'Top2_Confianza': float(top5_conf[1]),
            'Top3_Clase': int(top5_pred[2]),
            'Top3_Confianza': float(top5_conf[2]),
            'Top4_Clase': int(top5_pred[3]),
            'Top4_Confianza': float(top5_conf[3]),
            'Top5_Clase': int(top5_pred[4]),
            'Top5_Confianza': float(top5_conf[4])
        }
        
        resultados.append(resultado)
    
    df_resultados = pd.DataFrame(resultados)
    
    # Guardar reporte
    df_resultados.to_csv('reporte_predicciones_imagenes_personalizadas.csv', index=False)
    print("Reporte guardado como 'reporte_predicciones_imagenes_personalizadas.csv'")
    
    return df_resultados

def visualizar_predicciones(imagenes, rutas_imagenes, clases_predichas, confianzas, top5_indices, top5_confianzas, num_imagenes=20):
    """
    Visualiza las predicciones para un número específico de imágenes
    """
    num_imagenes = min(num_imagenes, len(imagenes))
    
    # Crear subplots
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(num_imagenes):
        # Mostrar imagen
        axes[i].imshow(imagenes[i])
        axes[i].axis('off')
        
        # Título con predicción
        clase_pred = clases_predichas[i]
        confianza = confianzas[i]
        nombre_archivo = os.path.basename(rutas_imagenes[i])
        
        axes[i].set_title(f'{nombre_archivo}\nClase: {clase_pred}\nConf: {confianza:.3f}', 
                         fontsize=8, pad=5)
    
    plt.tight_layout()
    plt.show()

def analizar_distribucion_clases(clases_predichas, confianzas):
    """
    Analiza la distribución de clases predichas
    """
    # Contar frecuencias de clases
    clases_unicas, frecuencias = np.unique(clases_predichas, return_counts=True)
    
    # Crear gráfico de barras
    plt.figure(figsize=(15, 6))
    plt.bar(clases_unicas, frecuencias)
    plt.title('Distribución de Clases Predichas')
    plt.xlabel('Clase')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Estadísticas de confianza
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(confianzas, bins=20, alpha=0.7, color='blue')
    plt.title('Distribución de Confianzas')
    plt.xlabel('Confianza')
    plt.ylabel('Frecuencia')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(confianzas)
    plt.title('Boxplot de Confianzas')
    plt.ylabel('Confianza')
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas resumidas
    print(f"\n=== ESTADÍSTICAS DE PREDICCIÓN ===")
    print(f"Total de imágenes: {len(clases_predichas)}")
    print(f"Clases únicas predichas: {len(clases_unicas)}")
    print(f"Confianza promedio: {np.mean(confianzas):.4f}")
    print(f"Confianza mediana: {np.median(confianzas):.4f}")
    print(f"Confianza mínima: {np.min(confianzas):.4f}")
    print(f"Confianza máxima: {np.max(confianzas):.4f}")
    
    # Top-5 clases más predichas
    indices_top5 = np.argsort(frecuencias)[-5:][::-1]
    print(f"\nTop-5 clases más predichas:")
    for i, idx in enumerate(indices_top5):
        print(f"{i+1}. Clase {clases_unicas[idx]}: {frecuencias[idx]} predicciones")

def evaluar_con_etiquetas_reales(modelo, imagenes, etiquetas_reales, rutas_imagenes):
    """
    Evalúa el modelo si se tienen etiquetas reales para las imágenes personalizadas
    """
    if not etiquetas_reales:
        print("No se proporcionaron etiquetas reales para evaluación")
        return
    
    # Predicciones
    clases_predichas, confianzas, _, _ = predecir_lote_imagenes(modelo, imagenes, rutas_imagenes)
    
    # Métricas de evaluación
    precision = np.mean(clases_predichas == etiquetas_reales)
    
    print(f"\n=== EVALUACIÓN CON ETIQUETAS REALES ===")
    print(f"Precisión: {precision:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(etiquetas_reales, clases_predichas)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión - Imágenes Personalizadas')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Etiqueta Predicha')
    plt.show()
    
    # Reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(etiquetas_reales, clases_predichas))

def main():
    """
    Función principal para evaluar imágenes personalizadas
    """
    print("=== EVALUADOR DE IMÁGENES PERSONALIZADAS ===")
    print("Clasificador de Autos - Dataset Cars196\n")
    
    # 1. Cargar modelo entrenado
    modelo = cargar_modelo_entrenado()
    if modelo is None:
        print("No se pudo cargar el modelo. Asegúrate de que existe 'modelo_autos_final.h5'")
        return
    
    # 2. Especificar carpeta con imágenes personalizadas
    carpeta_imagenes = input("Ingresa la ruta de la carpeta con tus 100 imágenes: ").strip()
    
    if not os.path.exists(carpeta_imagenes):
        print(f"La carpeta {carpeta_imagenes} no existe")
        return
    
    # 3. Cargar imágenes personalizadas
    imagenes, rutas_imagenes = cargar_imagenes_personalizadas(carpeta_imagenes)
    
    if not imagenes:
        print("No se pudieron cargar las imágenes")
        return
    
    print(f"Imágenes cargadas exitosamente: {len(imagenes)}")
    
    # 4. Realizar predicciones
    print("Realizando predicciones...")
    clases_predichas, confianzas, top5_indices, top5_confianzas = predecir_lote_imagenes(
        modelo, imagenes, rutas_imagenes
    )
    
    # 5. Crear reporte detallado
    df_resultados = crear_reporte_detallado(
        rutas_imagenes, clases_predichas, confianzas, top5_indices, top5_confianzas
    )
    
    # 6. Visualizar algunas predicciones
    print("Visualizando predicciones...")
    visualizar_predicciones(
        imagenes, rutas_imagenes, clases_predichas, confianzas, 
        top5_indices, top5_confianzas, num_imagenes=20
    )
    
    # 7. Analizar distribución de clases
    analizar_distribucion_clases(clases_predichas, confianzas)
    
    # 8. Preguntar si hay etiquetas reales
    tiene_etiquetas = input("\n¿Tienes etiquetas reales para estas imágenes? (s/n): ").lower().strip()
    
    if tiene_etiquetas == 's':
        print("Por favor, proporciona las etiquetas reales en el mismo orden que las imágenes")
        # Aquí podrías cargar las etiquetas desde un archivo o ingresarlas manualmente
        etiquetas_reales = []  # Cargar desde archivo o input manual
        evaluar_con_etiquetas_reales(modelo, imagenes, etiquetas_reales, rutas_imagenes)
    
    print(f"\n=== EVALUACIÓN COMPLETADA ===")
    print(f"Se evaluaron {len(imagenes)} imágenes personalizadas")
    print(f"Reporte guardado en: reporte_predicciones_imagenes_personalizadas.csv")

if __name__ == "__main__":
    main()
