# 🚗 Clasificador de Automóviles con Redes Neuronales Convolucionales

## 📋 Descripción del Proyecto

Este proyecto implementa un clasificador de imágenes utilizando Redes Neuronales Convolucionales (CNN) con PyTorch, específicamente diseñado para identificar automóviles y otros objetos del dataset CIFAR-10.

**Desarrollado para:** Especialización en Inteligencia Artificial - Universitaria Minuto de Dios  
**Asignatura:** Visión Por Computador - Semana 7  
**NRC:** 3664

## 👥 Integrantes del Grupo 3

- **David Stiven Benitez Guerra**
- **Leonardo Adolfo Mina Roldan**
- **David Orlando Timana Leyton**

## 🎯 Objetivos

- Implementar un clasificador CNN para automóviles usando PyTorch
- Evaluar el rendimiento en el dataset CIFAR-10
- Probar el modelo con imágenes personalizadas
- Analizar la efectividad de las CNNs en clasificación de vehículos

## 🏗️ Arquitectura del Modelo

### CNN Architecture
```
Input (32x32x3) 
    ↓
Conv2d(3→32, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(32→64, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(64→128, 3x3, padding=1) + ReLU + MaxPool2d(2x2)
    ↓
Flatten → Linear(128*4*4 → 256) + ReLU + Dropout(0.5)
    ↓
Linear(256 → 128) + ReLU + Dropout(0.5)
    ↓
Linear(128 → 10) → Softmax
```

### Hiperparámetros
- **Batch Size:** 64
- **Learning Rate:** 0.001
- **Épocas:** 10
- **Optimizador:** Adam
- **Función de Pérdida:** CrossEntropyLoss
- **Dropout:** 50%

## 📊 Resultados

### CIFAR-10 Dataset
- **Accuracy:** 74.62%
- **Clases:** 10 (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión)

### Dataset Personalizado (100 imágenes)
- **Accuracy:** 29.00%
- **Automóviles:** 100% (10/10 correctas)
- **Pájaros:** 80% (4/5 correctas)
- **Barcos:** 80% (8/10 correctas)
- **Aviones:** 40% (2/5 correctas)

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
PyTorch
torchvision
numpy
matplotlib
pandas
PIL (Pillow)
```

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/clasificador-autos-cnn.git
cd clasificador-autos-cnn

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install torch torchvision numpy matplotlib pandas pillow
```

### Ejecución

#### Para Google Colab
1. Sube `Clasificador_Autos_PyTorch_Colab_Final.py` a Google Colab
2. Configura GPU en Runtime → Change runtime type
3. Ejecuta el script completo

#### Para ejecución local
```bash
python Clasificador_Autos_PyTorch_Local_Simple.py
```

## 📁 Estructura del Proyecto

```
├── README.md                                    # Documentación principal
├── .gitignore                                  # Archivos a ignorar
├── requirements.txt                            # Dependencias
├── 
├── # Scripts principales
├── Clasificador_Autos_PyTorch_Colab_Final.py   # Versión para Google Colab
├── Clasificador_Autos_PyTorch_Local_Simple.py  # Versión local simplificada
├── Clasificador_Autos_PyTorch.py               # Versión local completa
├── 
├── # Scripts de utilidad
├── descargar_imagenes_prueba.py                # Descarga imágenes de prueba
├── probar_modelos_imagenes.py                  # Prueba modelos con imágenes
├── 
├── # Documentación
├── guion_video_clasificacion_automoviles-v1.txt # Guion video versión 1
├── guion_video_clasificacion_automoviles-v2.txt # Guion video versión 2
├── README_GoogleColab_PyTorch.md               # Instrucciones Colab
├── 
├── # Datasets y resultados
├── imagenes_prueba/                            # Imágenes de prueba (100)
├── resultados_predicciones_colab.csv           # Resultados de evaluación
└── modelo_autos_cnn_pytorch.pth                # Modelo entrenado
```

## 🔧 Características Técnicas

### Dataset
- **CIFAR-10:** 60,000 imágenes 32x32 píxeles
- **Fuente:** [Universidad de Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Clases:** 10 categorías diferentes

### Framework
- **PyTorch:** Framework principal de deep learning
- **Google Colab:** Entorno de desarrollo con GPU
- **Matplotlib:** Visualizaciones y gráficos

### Preprocesamiento
- Normalización RGB (media=0.5, std=0.5)
- Resize a 32x32 píxeles
- Data augmentation (en versiones avanzadas)

## 📈 Análisis de Resultados

### Fortalezas
- ✅ Excelente rendimiento en automóviles (100%)
- ✅ Arquitectura CNN efectiva para clasificación
- ✅ Implementación estable y reproducible
- ✅ Documentación completa

### Áreas de Mejora
- ⚠️ Gap de rendimiento entre CIFAR-10 y datos reales
- ⚠️ Necesidad de data augmentation
- ⚠️ Transfer learning para mejor generalización
- ⚠️ Más imágenes reales por clase

## 🎬 Video Presentación

El proyecto incluye guiones para video presentación:
- `guion_video_clasificacion_automoviles-v1.txt` - Versión técnica
- `guion_video_clasificacion_automoviles-v2.txt` - Versión más humana

## 📚 Referencias

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Google Colab](https://colab.research.google.com/)

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de la asignatura de Visión Por Computador. Las contribuciones están abiertas para mejoras y extensiones.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

---

**Desarrollado con ❤️ por el Grupo 3 - Especialización en IA - UNIMINUTO**
