# 🚗 Clasificador de Autos CNN - Google Colab

## 📋 Descripción

Este proyecto implementa una **Red Neuronal Convolucional (CNN)** para clasificación de automóviles utilizando el dataset **Cars196** en **Google Colab**. Está optimizado para ejecutarse en la nube con GPU gratuita.

## 🎯 Características Principales

- ✅ **Transfer Learning** con ResNet50V2 pre-entrenado
- ✅ **Data Augmentation** para mejor generalización
- ✅ **Fine-tuning** automático de las capas
- ✅ **GPU gratuita** de Google Colab
- ✅ **Subida de imágenes personalizadas** integrada
- ✅ **Evaluación completa** con métricas detalladas
- ✅ **Visualizaciones** de resultados

## 🚀 Cómo Usar en Google Colab

### **Paso 1: Abrir Google Colab**
1. Ve a [Google Colab](https://colab.research.google.com/)
2. Inicia sesión con tu cuenta de Google
3. Crea un nuevo notebook

### **Paso 2: Configurar GPU**
1. Ve a **Runtime** → **Change runtime type**
2. Selecciona **GPU** en Hardware accelerator
3. Haz clic en **Save**

### **Paso 3: Copiar el Código**
1. Copia todo el contenido del archivo `Clasificador_Autos_Cars196_GoogleColab.py`
2. Pégalo en una celda de Google Colab
3. Ejecuta la celda (Ctrl+Enter o Shift+Enter)

### **Paso 4: Ejecutar el Entrenamiento**
El script se ejecutará automáticamente y:
- 📥 Descargará el dataset Cars196 (~1.82 GB)
- 🏗️ Creará el modelo con transfer learning
- 🚀 Entrenará durante ~30-60 minutos
- 📊 Evaluará el modelo
- 📈 Mostrará gráficos de resultados

### **Paso 5: Evaluar Imágenes Personalizadas**
Al final del entrenamiento, el script te preguntará si quieres evaluar imágenes personalizadas:

1. **Preparar imágenes**: Comprime tus 100 imágenes en un archivo ZIP
2. **Subir archivo**: El script te permitirá subir el ZIP
3. **Ver resultados**: Obtendrás predicciones para cada imagen

## 📁 Estructura del Proyecto en Colab

```
/content/
├── Clasificador_Autos_Cars196_GoogleColab.py  # Script principal
├── modelo_autos_final.h5                      # Modelo entrenado
├── mejor_modelo_autos.h5                      # Mejor modelo durante entrenamiento
├── imagenes_personalizadas/                   # Carpeta con tus imágenes
│   ├── auto1.jpg
│   ├── auto2.jpg
│   └── ...
└── resultados_imagenes_personalizadas.csv     # Reporte de predicciones
```

## ⚙️ Configuración del Modelo

### **Hiperparámetros Optimizados:**
- **IMG_SIZE**: 224x224 (tamaño estándar para ResNet)
- **BATCH_SIZE**: 32 (balance memoria/velocidad)
- **EPOCHS**: 20 (entrenamiento inicial)
- **LEARNING_RATE**: 0.001 (Adam optimizer)
- **Fine-tuning**: 10 épocas adicionales

### **Arquitectura:**
```
ResNet50V2 (pre-entrenado en ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization + Dropout(0.5)
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
    ↓
Dense(256) + BatchNorm + Dropout(0.5)
    ↓
Dense(196, softmax)  # 196 clases de autos
```

## 📊 Resultados Esperados

Con la configuración actual, se esperan:
- **Accuracy**: 75-85% en conjunto de prueba
- **Top-5 Accuracy**: 90-95% en conjunto de prueba
- **Tiempo de entrenamiento**: 30-60 minutos (con GPU)

## 🔧 Funcionalidades Incluidas

### **1. Carga Automática del Dataset**
- Descarga automática de Cars196 desde TensorFlow Datasets
- Preprocesamiento automático de imágenes
- División train/validation/test

### **2. Transfer Learning Avanzado**
- ResNet50V2 pre-entrenado en ImageNet
- Fine-tuning automático de las últimas capas
- Callbacks para optimización (Early Stopping, ReduceLROnPlateau)

### **3. Data Augmentation**
- Flip horizontal aleatorio
- Ajuste de brillo (±20%)
- Ajuste de contraste (80%-120%)

### **4. Evaluación Completa**
- Métricas de precisión y Top-5 accuracy
- Visualización de curvas de entrenamiento
- Guardado automático del mejor modelo

### **5. Imágenes Personalizadas**
- Subida de archivos ZIP
- Predicciones en lote
- Reporte detallado en CSV
- Top-5 predicciones por imagen

## 📈 Visualizaciones Generadas

El script genera automáticamente:
1. **Gráfico de precisión** (entrenamiento vs validación)
2. **Gráfico de pérdida** (entrenamiento vs validación)
3. **Gráfico Top-5 precisión** (entrenamiento vs validación)
4. **Resumen de predicciones** para imágenes personalizadas

## 💾 Archivos Generados

### **Modelos:**
- `modelo_autos_final.h5`: Modelo final entrenado
- `mejor_modelo_autos.h5`: Mejor modelo durante entrenamiento

### **Resultados:**
- `resultados_imagenes_personalizadas.csv`: Reporte de predicciones

## 🎯 Ventajas de Google Colab

### **✅ Gratis:**
- GPU Tesla T4 gratuita
- 12GB de RAM
- 107GB de almacenamiento

### **✅ Fácil de usar:**
- No requiere instalación local
- Interfaz web intuitiva
- Integración con Google Drive

### **✅ Potente:**
- Acceso a GPU de alta calidad
- Entrenamiento rápido
- Resultados profesionales

## 🐛 Solución de Problemas

### **Error: "GPU not available"**
- Verifica que hayas seleccionado GPU en Runtime settings
- Reinicia el runtime si es necesario

### **Error: "Out of memory"**
- Reduce BATCH_SIZE a 16
- Reduce IMG_SIZE a 160x160

### **Error: "Dataset not found"**
- El script instala automáticamente tensorflow-datasets
- Verifica tu conexión a internet

### **Error: "Upload failed"**
- Asegúrate de que el archivo ZIP no sea muy grande
- Comprime las imágenes antes de subir

## 📚 Información del Dataset

### **Cars196:**
- **16,185 imágenes** de 196 clases de automóviles
- **8,144 imágenes** para entrenamiento
- **8,041 imágenes** para prueba
- **Clases**: Marca, Modelo, Año (ej: 2012 Tesla Model S)

## 🎉 ¡Listo para Usar!

1. **Copia el código** a Google Colab
2. **Configura GPU** en Runtime settings
3. **Ejecuta el script**
4. **Sube tus imágenes** cuando te lo solicite
5. **Analiza los resultados**

¡El clasificador estará listo para identificar automóviles en tus imágenes personalizadas!

---

**Autor**: David Timana  
**Curso**: Visión por Computador  
**Fecha**: 2024
