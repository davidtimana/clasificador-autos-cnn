# 🚗 GAN para Generación de Automóviles

## 📋 Descripción del Proyecto

Este proyecto implementa una **Red Generativa Adversaria (GAN)** usando PyTorch para generar imágenes de automóviles. Utiliza la arquitectura **DCGAN (Deep Convolutional GAN)** con el dataset CIFAR-10.

**Desarrollado para:** Especialización en Inteligencia Artificial - Visión por Computador  
**Tecnología:** PyTorch, DCGAN  
**Dataset:** CIFAR-10 (categoría automóviles)

## 🎯 Objetivos

- Implementar una GAN completa para generación de imágenes
- Generar 30 imágenes nuevas de automóviles
- Documentar el proceso de diseño e implementación
- Analizar la utilidad profesional de las GANs

## 🏗️ Arquitectura del Modelo

### Generador (Generator)
```
Input: Noise (100,) 
    ↓
Linear(100 → 256*4*4) + BatchNorm + ReLU
    ↓
ConvTranspose2d(256→128, 4x4, stride=2) + BatchNorm + ReLU
    ↓
ConvTranspose2d(128→64, 4x4, stride=2) + BatchNorm + ReLU
    ↓
ConvTranspose2d(64→32, 4x4, stride=2) + BatchNorm + ReLU
    ↓
ConvTranspose2d(32→3, 4x4, stride=2) + Tanh
    ↓
Output: Image (32x32x3)
```

### Discriminador (Discriminator)
```
Input: Image (32x32x3)
    ↓
Conv2d(3→32, 4x4, stride=2) + LeakyReLU
    ↓
Conv2d(32→64, 4x4, stride=2) + BatchNorm + LeakyReLU
    ↓
Conv2d(64→128, 4x4, stride=2) + BatchNorm + LeakyReLU
    ↓
Conv2d(128→256, 4x4, stride=2) + BatchNorm + LeakyReLU
    ↓
Flatten → Linear(256*2*2 → 1) + Sigmoid
    ↓
Output: Real/Fake probability
```

## ⚙️ Hiperparámetros

- **Batch Size:** 64
- **Latent Dimension:** 100
- **Learning Rate G:** 0.0002
- **Learning Rate D:** 0.0002
- **Épocas:** 100
- **Optimizador:** Adam (β1=0.5, β2=0.999)
- **Función de Pérdida:** Binary Cross Entropy

## 🚀 Instalación y Uso

### Prerrequisitos
```bash
Python 3.8+
PyTorch
torchvision
numpy
matplotlib
pillow
tqdm
```

### Instalación
```bash
# Navegar al directorio
cd gan_automoviles

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecución
```bash
# Ejecutar el script principal
python gan_automoviles.py
```

## 📁 Estructura del Proyecto

```
gan_automoviles/
├── gan_automoviles.py              # Script principal
├── requirements.txt                # Dependencias
├── README.md                       # Documentación
└── results/
    ├── generated_images/           # 30 imágenes generadas
    │   ├── generated_car_01.png
    │   ├── generated_car_02.png
    │   └── ...
    ├── training_progress.png       # Gráfica de entrenamiento
    ├── generator_final.pth         # Modelo final del generador
    └── discriminator_final.pth     # Modelo final del discriminador
```

## 📊 Resultados Esperados

### Imágenes Generadas
- **30 imágenes nuevas** de automóviles
- **Tamaño:** 32x32 píxeles
- **Formato:** PNG individual + grid completo

### Métricas de Entrenamiento
- **Pérdida del Generador:** Convergencia estable
- **Pérdida del Discriminador:** Balance con el generador
- **Calidad visual:** Automóviles reconocibles

## 🎯 Utilidad Profesional

### Aplicaciones Identificadas
1. **Data Augmentation:** Generar más automóviles para mejorar clasificadores
2. **Síntesis de Datos:** Crear datasets balanceados
3. **Investigación:** Explorar variaciones de diseño
4. **Educación:** Demostrar conceptos de IA generativa

### Necesidades que Soluciona
- **Escasez de datos:** Generar más ejemplos de automóviles
- **Desequilibrio de clases:** Balancear datasets
- **Variabilidad:** Crear nuevas variaciones
- **Investigación:** Explorar espacios latentes

## 🔬 Etapas del Desarrollo

### Etapa de Diseño
- ✅ Arquitectura DCGAN definida
- ✅ Hiperparámetros optimizados
- ✅ Dataset CIFAR-10 preparado
- ✅ Funciones de activación seleccionadas

### Etapa de Implementación
- ✅ Código completo en PyTorch
- ✅ Entrenamiento con visualizaciones
- ✅ Generación de 30 imágenes
- ✅ Análisis de resultados

## 📈 Características Técnicas

### Framework
- **PyTorch:** Framework principal
- **torchvision:** Transformaciones y utilidades
- **matplotlib:** Visualizaciones

### Preprocesamiento
- Normalización a [-1, 1]
- Batch Normalization
- Data augmentation implícita

### Regularización
- Batch Normalization
- Dropout (en capas densas)
- LeakyReLU para estabilidad

## 🎬 Video Presentación

El proyecto incluye documentación completa para presentación:
- Arquitectura detallada
- Proceso de implementación
- Resultados y análisis
- Utilidad profesional

## 📚 Referencias

- [DCGAN Paper](https://arxiv.org/abs/1511.06434)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte de la asignatura de Visión Por Computador. Las contribuciones están abiertas para mejoras y extensiones.

## 📄 Licencia

Este proyecto está bajo la Licencia MIT.

---

**Desarrollado con ❤️ por David Timana - Especialización en IA - UNIMINUTO**
