# -*- coding: utf-8 -*-
"""
Script para probar los modelos entrenados con 100 imágenes personalizadas
Autor: David Timana | Curso: Visión por Computador
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🧪 PROBADOR DE MODELOS CON IMÁGENES PERSONALIZADAS")
print("=" * 60)

# Configuración
IMAGES_DIR = "imagenes_prueba"
RESULTS_DIR = "resultados_pruebas"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Clases disponibles
classes = ['avion', 'automovil', 'pajaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion']
class_names_spanish = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

print(f"📂 Directorio de imágenes: {IMAGES_DIR}")
print(f"📊 Clases disponibles: {len(classes)}")
print(f"🎯 Objetivo: Probar modelos con 100 imágenes")

# =============================================================================
# 1. CARGAR Y PREPROCESAR IMÁGENES
# =============================================================================
print("\n1. Cargando y preprocesando imágenes...")

def load_and_preprocess_images():
    """Cargar y preprocesar todas las imágenes"""
    
    images = []
    labels = []
    filenames = []
    
    # Transformaciones para PyTorch
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Transformaciones para scikit-learn
    transform_sklearn = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(IMAGES_DIR, class_name)
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️  Directorio no encontrado: {class_dir}")
            continue
        
        # Obtener todas las imágenes en el directorio
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   📂 {class_name}: {len(image_files)} imágenes")
        
        for filename in image_files:
            try:
                filepath = os.path.join(class_dir, filename)
                
                # Cargar imagen
                img = Image.open(filepath).convert('RGB')
                
                # Preprocesar para PyTorch
                img_torch = transform(img)
                
                # Preprocesar para scikit-learn (MNIST style)
                img_sklearn = transform_sklearn(img)
                img_sklearn_flat = img_sklearn.view(-1).numpy()
                
                # Guardar datos
                images.append({
                    'torch': img_torch,
                    'sklearn': img_sklearn_flat,
                    'original': img
                })
                labels.append(class_idx)
                filenames.append(f"{class_name}/{filename}")
                
            except Exception as e:
                print(f"   ❌ Error procesando {filename}: {e}")
    
    print(f"✅ Total de imágenes cargadas: {len(images)}")
    return images, labels, filenames

# Cargar imágenes
images, labels, filenames = load_and_preprocess_images()

# =============================================================================
# 2. CARGAR MODELOS ENTRENADOS
# =============================================================================
print("\n2. Cargando modelos entrenados...")

def load_pytorch_model():
    """Cargar modelo PyTorch"""
    try:
        # Definir la arquitectura del modelo
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.fc1 = nn.Linear(128 * 4 * 4, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
                self.dropout = nn.Dropout(0.5)
                
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 128 * 4 * 4)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
        
        # Cargar modelo
        model = CNN()
        if os.path.exists('modelo_autos_cnn_pytorch.pth'):
            model.load_state_dict(torch.load('modelo_autos_cnn_pytorch.pth', map_location='cpu'))
            model.eval()
            print("   ✅ Modelo PyTorch cargado")
            return model
        else:
            print("   ❌ Archivo modelo_autos_cnn_pytorch.pth no encontrado")
            return None
    except Exception as e:
        print(f"   ❌ Error cargando modelo PyTorch: {e}")
        return None

def load_sklearn_model():
    """Cargar modelo scikit-learn"""
    try:
        if os.path.exists('modelo_sklearn.pkl'):
            model = joblib.load('modelo_sklearn.pkl')
            print("   ✅ Modelo scikit-learn cargado")
            return model
        else:
            print("   ❌ Archivo modelo_sklearn.pkl no encontrado")
            return None
    except Exception as e:
        print(f"   ❌ Error cargando modelo scikit-learn: {e}")
        return None

# Cargar modelos
pytorch_model = load_pytorch_model()
sklearn_model = load_sklearn_model()

# =============================================================================
# 3. HACER PREDICCIONES
# =============================================================================
print("\n3. Haciendo predicciones...")

def predict_with_pytorch(model, images):
    """Hacer predicciones con modelo PyTorch"""
    if model is None:
        return None
    
    predictions = []
    confidences = []
    
    model.eval()
    with torch.no_grad():
        for img_data in images:
            img_tensor = img_data['torch'].unsqueeze(0)
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predictions.append(predicted.item())
            confidences.append(confidence.item())
    
    return predictions, confidences

def predict_with_sklearn(model, images):
    """Hacer predicciones con modelo scikit-learn"""
    if model is None:
        return None
    
    predictions = []
    confidences = []
    
    for img_data in images:
        img_flat = img_data['sklearn'].reshape(1, -1)
        prediction = model.predict(img_flat)[0]
        confidence = model.predict_proba(img_flat).max()
        
        predictions.append(prediction)
        confidences.append(confidence)
    
    return predictions, confidences

# Hacer predicciones
print("   🔮 Prediciendo con PyTorch...")
pytorch_preds, pytorch_confs = predict_with_pytorch(pytorch_model, images)

print("   🔮 Prediciendo con scikit-learn...")
sklearn_preds, sklearn_confs = predict_with_sklearn(sklearn_model, images)

# =============================================================================
# 4. ANALIZAR RESULTADOS
# =============================================================================
print("\n4. Analizando resultados...")

def analyze_results():
    """Analizar y mostrar resultados"""
    
    results = []
    
    for i, (img_data, true_label, filename) in enumerate(zip(images, labels, filenames)):
        result = {
            'filename': filename,
            'true_class': class_names_spanish[true_label],
            'true_label': true_label
        }
        
        # Resultados PyTorch
        if pytorch_preds is not None:
            result['pytorch_pred'] = class_names_spanish[pytorch_preds[i]]
            result['pytorch_conf'] = pytorch_confs[i]
            result['pytorch_correct'] = pytorch_preds[i] == true_label
        
        # Resultados scikit-learn
        if sklearn_preds is not None:
            # Convertir predicción a entero si es string
            pred_idx = int(sklearn_preds[i]) if isinstance(sklearn_preds[i], str) else sklearn_preds[i]
            result['sklearn_pred'] = class_names_spanish[pred_idx]
            result['sklearn_conf'] = sklearn_confs[i]
            result['sklearn_correct'] = pred_idx == true_label
        
        results.append(result)
    
    return results

results = analyze_results()

# =============================================================================
# 5. CALCULAR MÉTRICAS
# =============================================================================
print("\n5. Calculando métricas...")

def calculate_metrics():
    """Calcular métricas de rendimiento"""
    
    metrics = {}
    
    # Métricas PyTorch
    if pytorch_preds is not None:
        pytorch_correct = sum(1 for i, pred in enumerate(pytorch_preds) if pred == labels[i])
        pytorch_accuracy = pytorch_correct / len(labels) * 100
        pytorch_avg_conf = np.mean(pytorch_confs) * 100
        
        metrics['pytorch'] = {
            'accuracy': pytorch_accuracy,
            'correct': pytorch_correct,
            'total': len(labels),
            'avg_confidence': pytorch_avg_conf
        }
        
        print(f"   📊 PyTorch:")
        print(f"      ✅ Accuracy: {pytorch_accuracy:.2f}%")
        print(f"      ✅ Correctas: {pytorch_correct}/{len(labels)}")
        print(f"      ✅ Confianza promedio: {pytorch_avg_conf:.2f}%")
    
    # Métricas scikit-learn
    if sklearn_preds is not None:
        sklearn_correct = sum(1 for i, pred in enumerate(sklearn_preds) if int(pred) == labels[i])
        sklearn_accuracy = sklearn_correct / len(labels) * 100
        sklearn_avg_conf = np.mean(sklearn_confs) * 100
        
        metrics['sklearn'] = {
            'accuracy': sklearn_accuracy,
            'correct': sklearn_correct,
            'total': len(labels),
            'avg_confidence': sklearn_avg_conf
        }
        
        print(f"   📊 scikit-learn:")
        print(f"      ✅ Accuracy: {sklearn_accuracy:.2f}%")
        print(f"      ✅ Correctas: {sklearn_correct}/{len(labels)}")
        print(f"      ✅ Confianza promedio: {sklearn_avg_conf:.2f}%")
    
    return metrics

metrics = calculate_metrics()

# =============================================================================
# 6. ANÁLISIS POR CLASE
# =============================================================================
print("\n6. Análisis por clase...")

def analyze_by_class():
    """Analizar rendimiento por clase"""
    
    class_analysis = {}
    
    for class_idx, class_name in enumerate(classes):
        class_indices = [i for i, label in enumerate(labels) if label == class_idx]
        
        if not class_indices:
            continue
        
        analysis = {
            'class_name': class_names_spanish[class_idx],
            'total_images': len(class_indices)
        }
        
        # PyTorch por clase
        if pytorch_preds is not None:
            pytorch_class_correct = sum(1 for i in class_indices if pytorch_preds[i] == class_idx)
            pytorch_class_accuracy = pytorch_class_correct / len(class_indices) * 100
            analysis['pytorch_accuracy'] = pytorch_class_accuracy
            analysis['pytorch_correct'] = pytorch_class_correct
        
        # scikit-learn por clase
        if sklearn_preds is not None:
            sklearn_class_correct = sum(1 for i in class_indices if int(sklearn_preds[i]) == class_idx)
            sklearn_class_accuracy = sklearn_class_correct / len(class_indices) * 100
            analysis['sklearn_accuracy'] = sklearn_class_accuracy
            analysis['sklearn_correct'] = sklearn_class_correct
        
        class_analysis[class_name] = analysis
        
        print(f"   📂 {class_names_spanish[class_idx]}:")
        print(f"      📊 Total: {len(class_indices)} imágenes")
        if pytorch_preds is not None:
            print(f"      🧠 PyTorch: {pytorch_class_accuracy:.1f}% ({pytorch_class_correct}/{len(class_indices)})")
        if sklearn_preds is not None:
            print(f"      🔬 scikit-learn: {sklearn_class_accuracy:.1f}% ({sklearn_class_correct}/{len(class_indices)})")
    
    return class_analysis

class_analysis = analyze_by_class()

# =============================================================================
# 7. GUARDAR RESULTADOS
# =============================================================================
print("\n7. Guardando resultados...")

# Guardar resultados en CSV
df_results = pd.DataFrame(results)
csv_path = os.path.join(RESULTS_DIR, 'resultados_predicciones.csv')
df_results.to_csv(csv_path, index=False)
print(f"   ✅ Resultados guardados en: {csv_path}")

# Guardar métricas en CSV
df_metrics = pd.DataFrame([metrics])
metrics_path = os.path.join(RESULTS_DIR, 'metricas_generales.csv')
df_metrics.to_csv(metrics_path, index=False)
print(f"   ✅ Métricas guardadas en: {metrics_path}")

# Guardar análisis por clase
df_class_analysis = pd.DataFrame(class_analysis).T
class_analysis_path = os.path.join(RESULTS_DIR, 'analisis_por_clase.csv')
df_class_analysis.to_csv(class_analysis_path)
print(f"   ✅ Análisis por clase guardado en: {class_analysis_path}")

# =============================================================================
# 8. VISUALIZAR RESULTADOS
# =============================================================================
print("\n8. Visualizando resultados...")

def visualize_results():
    """Crear visualizaciones de los resultados"""
    
    # Gráfico de comparación de accuracy
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: Comparación general
    plt.subplot(2, 2, 1)
    models = []
    accuracies = []
    
    if 'pytorch' in metrics:
        models.append('PyTorch')
        accuracies.append(metrics['pytorch']['accuracy'])
    
    if 'sklearn' in metrics:
        models.append('scikit-learn')
        accuracies.append(metrics['sklearn']['accuracy'])
    
    bars = plt.bar(models, accuracies, color=['blue', 'orange'])
    plt.title('Accuracy General por Modelo')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Agregar valores en las barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Subplot 2: Accuracy por clase (PyTorch)
    if 'pytorch' in metrics:
        plt.subplot(2, 2, 2)
        class_names = [class_analysis[cls]['class_name'] for cls in classes if cls in class_analysis]
        class_accuracies = [class_analysis[cls]['pytorch_accuracy'] for cls in classes if cls in class_analysis]
        
        bars = plt.bar(range(len(class_names)), class_accuracies, color='blue', alpha=0.7)
        plt.title('Accuracy por Clase - PyTorch')
        plt.xlabel('Clase')
        plt.ylabel('Accuracy (%)')
        plt.xticks(range(len(class_names)), [name[:5] for name in class_names], rotation=45)
        plt.ylim(0, 100)
    
    # Subplot 3: Accuracy por clase (scikit-learn)
    if 'sklearn' in metrics:
        plt.subplot(2, 2, 3)
        class_names = [class_analysis[cls]['class_name'] for cls in classes if cls in class_analysis]
        class_accuracies = [class_analysis[cls]['sklearn_accuracy'] for cls in classes if cls in class_analysis]
        
        bars = plt.bar(range(len(class_names)), class_accuracies, color='orange', alpha=0.7)
        plt.title('Accuracy por Clase - scikit-learn')
        plt.xlabel('Clase')
        plt.ylabel('Accuracy (%)')
        plt.xticks(range(len(class_names)), [name[:5] for name in class_names], rotation=45)
        plt.ylim(0, 100)
    
    # Subplot 4: Ejemplos de predicciones
    plt.subplot(2, 2, 4)
    
    # Mostrar algunas imágenes con sus predicciones
    sample_indices = np.random.choice(len(results), min(6, len(results)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        result = results[idx]
        img_data = images[idx]
        
        plt.subplot(2, 3, i+1)
        plt.imshow(img_data['original'])
        plt.title(f'Real: {result["true_class"][:5]}\nPred: {result.get("pytorch_pred", "N/A")[:5]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'resultados_visualizacion.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Visualizaciones guardadas")

visualize_results()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 60)
print("🎉 PRUEBAS COMPLETADAS")
print("=" * 60)
print(f"📊 Total de imágenes probadas: {len(images)}")
print(f"📂 Resultados guardados en: {RESULTS_DIR}")

if 'pytorch' in metrics:
    print(f"🧠 PyTorch - Accuracy: {metrics['pytorch']['accuracy']:.2f}%")

if 'sklearn' in metrics:
    print(f"🔬 scikit-learn - Accuracy: {metrics['sklearn']['accuracy']:.2f}%")

print(f"\n📁 Archivos generados:")
print(f"   📄 resultados_predicciones.csv - Predicciones detalladas")
print(f"   📄 metricas_generales.csv - Métricas generales")
print(f"   📄 analisis_por_clase.csv - Análisis por clase")
print(f"   🖼️  resultados_visualizacion.png - Gráficos")

print(f"\n✅ ¡Análisis completo realizado!")
print("🚀 ¡Los modelos han sido probados exitosamente con tus 100 imágenes!")
