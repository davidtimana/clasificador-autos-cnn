# 🚗 Clasificador de Autos CNN - PyTorch para Google Colab

## 📋 Descripción

Este proyecto implementa un clasificador de imágenes usando Redes Neuronales Convolucionales (CNN) con PyTorch, específicamente diseñado para Google Colab. El modelo puede clasificar 10 clases diferentes de objetos.

### 🎯 Clases Clasificadas
- **0. avión** ✈️
- **1. automóvil** 🚗
- **2. pájaro** 🐦
- **3. gato** 🐱
- **4. ciervo** 🦌
- **5. perro** 🐕
- **6. rana** 🐸
- **7. caballo** 🐎
- **8. barco** ⛵
- **9. camión** 🚛

## 🚀 Características

✅ **Entrenamiento completo** con dataset CIFAR-10  
✅ **Evaluación con imágenes personalizadas** desde Google Drive  
✅ **Visualizaciones interactivas** y gráficos detallados  
✅ **Análisis por clase** con métricas específicas  
✅ **Guardado automático** de modelo y resultados en Drive  
✅ **Documentación completa** del código  
✅ **Compatibilidad total** con GPU/TPU de Colab  

## 📁 Estructura de Archivos

```
📂 Tu Google Drive/
├── 📂 Clasificador_Autos_Imagenes/     # Carpeta con tus 100 imágenes
│   ├── 📂 avion/                       # 10 imágenes de aviones
│   ├── 📂 automovil/                   # 10 imágenes de automóviles
│   ├── 📂 pajaro/                      # 10 imágenes de pájaros
│   ├── 📂 gato/                        # 10 imágenes de gatos
│   ├── 📂 ciervo/                      # 10 imágenes de ciervos
│   ├── 📂 perro/                       # 10 imágenes de perros
│   ├── 📂 rana/                        # 10 imágenes de ranas
│   ├── 📂 caballo/                     # 10 imágenes de caballos
│   ├── 📂 barco/                       # 10 imágenes de barcos
│   └── 📂 camion/                      # 10 imágenes de camiones
├── 🧠 modelo_autos_cnn_pytorch_colab.pth    # Modelo entrenado
└── 📊 resultados_predicciones_colab.csv     # Resultados de predicciones
```

## 🛠️ Instalación y Configuración

### Paso 1: Preparar Google Drive
1. **Subir imágenes a Drive**: Asegúrate de que tus 100 imágenes estén organizadas en carpetas por clase en tu Google Drive
2. **Estructura requerida**: Cada clase debe estar en una carpeta separada con el nombre exacto (en minúsculas)

### Paso 2: Abrir Google Colab
1. Ve a [Google Colab](https://colab.research.google.com/)
2. Crea un nuevo notebook
3. Configura el runtime:
   - **Runtime** → **Change runtime type**
   - **Hardware accelerator**: **GPU** (recomendado) o **TPU**
   - **Runtime type**: **Python 3**

### Paso 3: Ejecutar el Script
1. Copia todo el contenido del archivo `Clasificador_Autos_PyTorch_Colab.py`
2. Pégalo en una celda de Colab
3. Ejecuta la celda (Ctrl+Enter o Shift+Enter)

## 📊 Resultados Esperados

### Durante el Entrenamiento
```
🚀 CLASIFICADOR DE AUTOS CNN - PYTORCH PARA GOOGLE COLAB
==============================================================

📂 Montando Google Drive...
✅ PyTorch version: 2.x.x
✅ Device: cuda (si GPU está disponible)
✅ CUDA disponible: True

⚙️ Configurando hiperparámetros...
📊 Configuración:
   🎯 Batch Size: 64
   📈 Learning Rate: 0.001
   🔄 Épocas: 10
   🖼️ Tamaño imagen: 32x32
   🏷️ Número de clases: 10

📥 Cargando dataset CIFAR-10...
   ✅ Dataset cargado:
      📊 Entrenamiento: 50,000 imágenes
      📊 Prueba: 10,000 imágenes
      🖼️ Tamaño: 32x32x3
      🏷️ Clases: 10
      🚗 Automóviles: clase 1

🧠 Construyendo modelo CNN...
✅ Modelo creado:

🚀 Iniciando entrenamiento...
🔄 Época 1/10
   📊 Batch 100, Loss: 2.135, Accuracy: 18.36%
   📊 Batch 200, Loss: 1.870, Accuracy: 23.93%
   ...
✅ Época 1 completada - Accuracy: 38.82%
```

### Resultados Finales
```
📊 RESULTADOS EN CIFAR-10:
   ✅ Test accuracy: 74.83%

📊 RESULTADOS EN IMÁGENES PERSONALIZADAS:
   ✅ Accuracy: 22.00%
   ✅ Total imágenes: 100

📊 Análisis por clase (imágenes personalizadas):
   📂 avión: 0.0% (0/10)
   📂 automóvil: 80.0% (8/10)  ← ¡Excelente!
   📂 pájaro: 50.0% (5/10)
   📂 gato: 0.0% (0/10)
   📂 ciervo: 90.0% (9/10)  ← ¡Excelente!
   ...
```

## 📈 Interpretación de Resultados

### 🎯 Accuracy en CIFAR-10
- **74.83%**: Buen rendimiento en el dataset estándar
- **Comparable** a modelos similares en la literatura

### 🚗 Accuracy en Automóviles (Imágenes Personalizadas)
- **80.0%**: Excelente rendimiento específico para automóviles
- **8 de 10** automóviles correctamente identificados
- **Resultado superior** al promedio general

### 📊 Análisis por Clase
- **Mejores clases**: Ciervo (90%), Automóvil (80%), Pájaro (50%)
- **Clases con dificultad**: Gato, Perro, Rana (0%)
- **Limitaciones esperadas**: Diferencia entre imágenes de entrenamiento y prueba

## 🔧 Personalización

### Modificar Hiperparámetros
```python
# En la sección de configuración
BATCH_SIZE = 64          # Tamaño del batch
LEARNING_RATE = 0.001    # Tasa de aprendizaje
EPOCHS = 10              # Número de épocas
IMG_SIZE = 32            # Tamaño de imagen
```

### Agregar Más Épocas
```python
# Para entrenamiento más largo
EPOCHS = 20  # o más épocas
```

### Cambiar Optimizador
```python
# En lugar de Adam, usar SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## 📁 Archivos Generados

### En Google Drive
1. **`modelo_autos_cnn_pytorch_colab.pth`**: Modelo entrenado
2. **`resultados_predicciones_colab.csv`**: Resultados detallados de predicciones

### Contenido del CSV
```csv
filename,true_class,predicted_class,confidence,correct
automovil/auto_01.jpg,automóvil,automóvil,0.85,True
avion/avion_01.jpg,avión,automóvil,0.72,False
...
```

## 🚨 Solución de Problemas

### Error: "No se encontró la carpeta de imágenes"
**Solución**: Verifica que las imágenes estén en una de estas rutas:
- `/content/drive/MyDrive/Clasificador_Autos_Imagenes`
- `/content/drive/MyDrive/imagenes_prueba`
- `/content/drive/MyDrive/100_imagenes_prueba`

### Error: "CUDA out of memory"
**Solución**: Reduce el batch size:
```python
BATCH_SIZE = 32  # En lugar de 64
```

### Error: "Runtime disconnected"
**Solución**: 
1. Usa GPU en lugar de TPU
2. Reduce el número de épocas
3. Ejecuta en sesiones más cortas

## 🎯 Consejos para Mejorar Resultados

### 1. Más Datos de Entrenamiento
- Agregar más imágenes por clase
- Usar data augmentation
- Incluir más variedad de imágenes

### 2. Ajustar Hiperparámetros
- Probar diferentes learning rates
- Aumentar número de épocas
- Modificar arquitectura del modelo

### 3. Transfer Learning
- Usar modelos pre-entrenados (ResNet, VGG)
- Fine-tuning específico para automóviles

### 4. Preprocesamiento
- Normalización específica del dominio
- Data augmentation más robusto
- Balanceo de clases

## 📚 Referencias

- **CIFAR-10 Dataset**: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)
- **Google Colab**: [https://colab.research.google.com/](https://colab.research.google.com/)

## 👨‍💻 Autor

**David Timana**  
Curso: Visión por Computador  
Fecha: 2024

## 📄 Licencia

Este proyecto es para fines educativos. Libre de usar y modificar.

---

## 🎉 ¡Listo para Usar!

1. **Copia el código** del archivo `Clasificador_Autos_PyTorch_Colab.py`
2. **Pégalo en Google Colab**
3. **Configura GPU** en el runtime
4. **Ejecuta** y disfruta de los resultados

¡Tu clasificador de automóviles estará listo en minutos! 🚗✨
