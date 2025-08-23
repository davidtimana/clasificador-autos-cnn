# -*- coding: utf-8 -*-
"""
Script para Google Colab - Probar modelos con imágenes desde Google Drive
Autor: David Timana | Curso: Visión por Computador
"""

# =============================================================================
# CONFIGURACIÓN PARA GOOGLE COLAB
# =============================================================================
print("🚀 CONFIGURACIÓN PARA GOOGLE COLAB")
print("=" * 50)

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DESCARGAR IMÁGENES DESDE GOOGLE DRIVE
# =============================================================================
print("\n1. Descargando imágenes desde Google Drive...")

# ID de la carpeta compartida
FOLDER_ID = "1CGxjT93JInXM1FNHFBEb8zM3U5avQ6qi"

# Crear directorio local
IMAGES_DIR = "imagenes_drive"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# Clases disponibles
classes = ['avion', 'automovil', 'pajaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camion']
class_names_spanish = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

def download_from_drive():
    """Descargar imágenes desde Google Drive"""
    
    from google.colab import files
    import zipfile
    
    # Crear subdirectorios
    for class_name in classes:
        class_dir = os.path.join(IMAGES_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    print("📥 Descargando desde Google Drive...")
    print("💡 Instrucciones:")
    print("   1. Ve a tu Google Drive")
    print("   2. Selecciona todas las carpetas de imágenes")
    print("   3. Haz clic derecho -> 'Descargar'")
    print("   4. Sube el archivo ZIP aquí")
    
    # Esperar a que el usuario suba el archivo
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            print(f"📦 Extrayendo {filename}...")
            
            # Extraer archivo ZIP
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(IMAGES_DIR)
            
            print("✅ Archivo extraído exitosamente")
            break
    
    # Verificar imágenes descargadas
    total_images = 0
    for class_name in classes:
        class_dir = os.path.join(IMAGES_DIR, class_name)
        if os.path.exists(class_dir):
            images_in_class = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            print(f"   📂 {class_name}: {images_in_class} imágenes")
            total_images += images_in_class
    
    print(f"✅ Total de imágenes descargadas: {total_images}")
    return total_images

# =============================================================================
# 2. ALTERNATIVA: USAR RUTA DIRECTA DE DRIVE
# =============================================================================
def use_drive_path():
    """Usar ruta directa de Google Drive"""
    
    # Ruta en Google Drive (ajustar según tu estructura)
    drive_path = "/content/drive/MyDrive/Clasificador_Autos_Imagenes"
    
    if os.path.exists(drive_path):
        print(f"📂 Usando ruta de Drive: {drive_path}")
        
        # Crear enlaces simbólicos o copiar archivos
        for class_name in classes:
            source_dir = os.path.join(drive_path, class_name)
            target_dir = os.path.join(IMAGES_DIR, class_name)
            
            if os.path.exists(source_dir):
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                # Copiar archivos
                import shutil
                for file in os.listdir(source_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        shutil.copy2(
                            os.path.join(source_dir, file),
                            os.path.join(target_dir, file)
                        )
        
        return True
    else:
        print(f"❌ Ruta no encontrada: {drive_path}")
        return False

# Intentar usar Drive directamente, si no funciona, descargar
if not use_drive_path():
    download_from_drive()

# =============================================================================
# 3. CARGAR Y PREPROCESAR IMÁGENES
# =============================================================================
print("\n2. Cargando y preprocesando imágenes...")

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
                
                # Guardar datos
                images.append({
                    'torch': img_torch,
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
# 4. CARGAR MODELO PYTORCH (ENTRENADO LOCALMENTE)
# =============================================================================
print("\n3. Cargando modelo PyTorch...")

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
        
        # Intentar cargar desde Drive
        model_paths = [
            "/content/drive/MyDrive/modelo_autos_cnn_pytorch.pth",
            "modelo_autos_cnn_pytorch.pth"
        ]
        
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = CNN()
                model.load_state_dict(torch.load(path, map_location='cpu'))
                model.eval()
                print(f"   ✅ Modelo PyTorch cargado desde: {path}")
                break
        
        if model is None:
            print("   ❌ Modelo PyTorch no encontrado")
            print("   💡 Sube el archivo 'modelo_autos_cnn_pytorch.pth' a tu Drive")
            return None
            
        return model
    except Exception as e:
        print(f"   ❌ Error cargando modelo PyTorch: {e}")
        return None

# Cargar modelo
pytorch_model = load_pytorch_model()

# =============================================================================
# 5. HACER PREDICCIONES
# =============================================================================
print("\n4. Haciendo predicciones...")

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

# Hacer predicciones
if pytorch_model is not None:
    print("   🔮 Prediciendo con PyTorch...")
    pytorch_preds, pytorch_confs = predict_with_pytorch(pytorch_model, images)
else:
    pytorch_preds, pytorch_confs = None, None

# =============================================================================
# 6. ANALIZAR RESULTADOS
# =============================================================================
print("\n5. Analizando resultados...")

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
        
        results.append(result)
    
    return results

results = analyze_results()

# =============================================================================
# 7. CALCULAR MÉTRICAS
# =============================================================================
print("\n6. Calculando métricas...")

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
        
        print(f"📊 PyTorch:")
        print(f"   ✅ Accuracy: {pytorch_accuracy:.2f}%")
        print(f"   ✅ Correctas: {pytorch_correct}/{len(labels)}")
        print(f"   ✅ Confianza promedio: {pytorch_avg_conf:.2f}%")
    
    return metrics

metrics = calculate_metrics()

# =============================================================================
# 8. ANÁLISIS POR CLASE
# =============================================================================
print("\n7. Análisis por clase...")

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
        
        class_analysis[class_name] = analysis
        
        print(f"   📂 {class_names_spanish[class_idx]}:")
        print(f"      📊 Total: {len(class_indices)} imágenes")
        if pytorch_preds is not None:
            print(f"      🧠 PyTorch: {pytorch_class_accuracy:.1f}% ({pytorch_class_correct}/{len(class_indices)})")
    
    return class_analysis

class_analysis = analyze_by_class()

# =============================================================================
# 9. VISUALIZAR RESULTADOS
# =============================================================================
print("\n8. Visualizando resultados...")

def visualize_results():
    """Crear visualizaciones de los resultados"""
    
    if pytorch_preds is None:
        print("   ⚠️  No hay predicciones para visualizar")
        return
    
    # Gráfico de accuracy por clase
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Accuracy por clase
    plt.subplot(1, 2, 1)
    class_names = [class_analysis[cls]['class_name'] for cls in classes if cls in class_analysis]
    class_accuracies = [class_analysis[cls]['pytorch_accuracy'] for cls in classes if cls in class_analysis]
    
    bars = plt.bar(range(len(class_names)), class_accuracies, color='blue', alpha=0.7)
    plt.title('Accuracy por Clase - PyTorch')
    plt.xlabel('Clase')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(len(class_names)), [name[:5] for name in class_names], rotation=45)
    plt.ylim(0, 100)
    
    # Agregar valores en las barras
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Ejemplos de predicciones
    plt.subplot(1, 2, 2)
    
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
    plt.show()
    
    print("   ✅ Visualizaciones completadas")

visualize_results()

# =============================================================================
# 10. GUARDAR RESULTADOS EN DRIVE
# =============================================================================
print("\n9. Guardando resultados en Drive...")

def save_results_to_drive():
    """Guardar resultados en Google Drive"""
    
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(results)
    
    # Guardar en Drive
    drive_results_path = "/content/drive/MyDrive/resultados_colab.csv"
    df_results.to_csv(drive_results_path, index=False)
    
    print(f"   ✅ Resultados guardados en Drive: {drive_results_path}")
    
    # Mostrar algunas predicciones
    print("\n📊 Ejemplos de predicciones:")
    for i, result in enumerate(results[:5]):
        print(f"   {i+1}. {result['filename']}")
        print(f"      Real: {result['true_class']}")
        if 'pytorch_pred' in result:
            print(f"      Pred: {result['pytorch_pred']} (conf: {result['pytorch_conf']:.2f})")
        print()

save_results_to_drive()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 60)
print("🎉 PRUEBAS EN COLAB COMPLETADAS")
print("=" * 60)
print(f"📊 Total de imágenes probadas: {len(images)}")
print(f"📂 Imágenes desde: Google Drive")

if 'pytorch' in metrics:
    print(f"🧠 PyTorch - Accuracy: {metrics['pytorch']['accuracy']:.2f}%")

print(f"\n📁 Resultados guardados en Google Drive")
print(f"✅ ¡Análisis completo realizado en Colab!")

# =============================================================================
# INSTRUCCIONES PARA EL USUARIO
# =============================================================================
print(f"\n📋 INSTRUCCIONES:")
print(f"1. 📂 Las imágenes están en tu Google Drive")
print(f"2. 🧠 Sube el modelo 'modelo_autos_cnn_pytorch.pth' a tu Drive")
print(f"3. 📊 Los resultados se guardan automáticamente en Drive")
print(f"4. 🔄 Puedes ejecutar este script múltiples veces")
