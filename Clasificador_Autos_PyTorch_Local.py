# -*- coding: utf-8 -*-
"""
===============================================================================
CLASIFICADOR DE AUTOS CNN - VERSIÓN PYTORCH PARA EJECUCIÓN LOCAL
===============================================================================

DESCRIPCIÓN:
    Este script implementa un clasificador de imágenes usando Redes Neuronales
    Convolucionales (CNN) con PyTorch, adaptado para ejecución local en macOS.
    
    El modelo puede clasificar 10 clases diferentes:
    - avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión

CARACTERÍSTICAS:
    ✅ Entrenamiento completo con CIFAR-10
    ✅ Evaluación con imágenes personalizadas desde carpeta local
    ✅ Visualizaciones interactivas
    ✅ Análisis detallado por clase
    ✅ Guardado automático de resultados
    ✅ Compatible con macOS (sin multiprocessing)

AUTOR: David Timana
CURSO: Visión por Computador
FECHA: 2024
===============================================================================
"""

# =============================================================================
# 1. CONFIGURACIÓN INICIAL
# =============================================================================
print("🚀 CLASIFICADOR DE AUTOS CNN - PYTORCH PARA EJECUCIÓN LOCAL")
print("=" * 70)

# Importaciones necesarias
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Verificar configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Device: {device}")
print(f"✅ CUDA disponible: {torch.cuda.is_available()}")

# =============================================================================
# 2. CONFIGURACIÓN DE HIPERPARÁMETROS
# =============================================================================
print("\n⚙️ Configurando hiperparámetros...")

# Hiperparámetros del modelo
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = 32
NUM_CLASSES = 10

# Clases disponibles
classes = ('avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
           'perro', 'rana', 'caballo', 'barco', 'camión')

print(f"📊 Configuración:")
print(f"   🎯 Batch Size: {BATCH_SIZE}")
print(f"   📈 Learning Rate: {LEARNING_RATE}")
print(f"   🔄 Épocas: {EPOCHS}")
print(f"   🖼️ Tamaño imagen: {IMG_SIZE}x{IMG_SIZE}")
print(f"   🏷️ Número de clases: {NUM_CLASSES}")

# =============================================================================
# 3. CARGAR Y PREPROCESAR DATASET CIFAR-10
# =============================================================================
print("\n📥 Cargando dataset CIFAR-10...")

def load_cifar10_dataset():
    """
    Cargar y preprocesar el dataset CIFAR-10
    
    Returns:
        tuple: (trainloader, testloader, trainset, testset)
    """
    
    # Transformaciones para preprocesamiento
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Cargar datasets
    print("   📥 Descargando CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Crear dataloaders (sin multiprocessing para compatibilidad con macOS)
    trainloader = DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0  # Importante: 0 para macOS
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0  # Importante: 0 para macOS
    )
    
    print(f"   ✅ Dataset cargado:")
    print(f"      📊 Entrenamiento: {len(trainset):,} imágenes")
    print(f"      📊 Prueba: {len(testset):,} imágenes")
    print(f"      🖼️ Tamaño: {IMG_SIZE}x{IMG_SIZE}x3")
    print(f"      🏷️ Clases: {NUM_CLASSES}")
    print(f"      🚗 Automóviles: clase 1")
    
    return trainloader, testloader, trainset, testset

# Cargar dataset
trainloader, testloader, trainset, testset = load_cifar10_dataset()

# =============================================================================
# 4. DEFINIR ARQUITECTURA DEL MODELO CNN
# =============================================================================
print("\n🧠 Construyendo modelo CNN...")

class CNN(nn.Module):
    """
    Arquitectura de Red Neuronal Convolucional (CNN)
    
    Capas:
    - 3 capas convolucionales con MaxPooling
    - 3 capas densas con Dropout
    - Activación ReLU
    - Salida Softmax para clasificación
    """
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # Primera capa convolucional
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Segunda capa convolucional
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Tercera capa convolucional
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # Capas densas (fully connected)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass del modelo
        
        Args:
            x: Tensor de entrada (batch_size, 3, 32, 32)
            
        Returns:
            Tensor de salida (batch_size, num_classes)
        """
        # Primera capa convolucional: 32x32 -> 16x16
        x = self.pool(self.relu(self.conv1(x)))
        
        # Segunda capa convolucional: 16x16 -> 8x8
        x = self.pool(self.relu(self.conv2(x)))
        
        # Tercera capa convolucional: 8x8 -> 4x4
        x = self.pool(self.relu(self.conv3(x)))
        
        # Aplanar para capas densas
        x = x.view(-1, 128 * 4 * 4)
        
        # Capas densas con dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Crear modelo
model = CNN().to(device)
print("✅ Modelo creado:")
print(model)

# =============================================================================
# 5. CONFIGURAR ENTRENAMIENTO
# =============================================================================
print("\n⚙️ Configurando entrenamiento...")

# Función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"✅ Configuración de entrenamiento:")
print(f"   📉 Función de pérdida: CrossEntropyLoss")
print(f"   🎯 Optimizador: Adam (lr={LEARNING_RATE})")
print(f"   💻 Device: {device}")

# =============================================================================
# 6. FUNCIÓN DE ENTRENAMIENTO
# =============================================================================
print("\n🚀 Iniciando entrenamiento...")

def train_model(model, trainloader, epochs=EPOCHS):
    """
    Entrenar el modelo CNN
    
    Args:
        model: Modelo PyTorch
        trainloader: DataLoader para entrenamiento
        epochs: Número de épocas
        
    Returns:
        tuple: (train_losses, train_accuracies)
    """
    
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"🔄 Época {epoch+1}/{epochs}")
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mostrar progreso cada 100 batches
            if i % 100 == 99:
                print(f'   📊 Batch {i+1}, Loss: {running_loss/100:.3f}, Accuracy: {100*correct/total:.2f}%')
                running_loss = 0.0
        
        # Calcular accuracy de la época
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
        train_losses.append(running_loss / len(trainloader))
        
        print(f"✅ Época {epoch+1} completada - Accuracy: {epoch_accuracy:.2f}%")
    
    return train_losses, train_accuracies

# Entrenar modelo
train_losses, train_accuracies = train_model(model, trainloader)
print("✅ Entrenamiento completado!")

# =============================================================================
# 7. EVALUAR MODELO EN CIFAR-10
# =============================================================================
print("\n📊 Evaluando modelo en CIFAR-10...")

def evaluate_model(model, testloader):
    """
    Evaluar el modelo en el conjunto de prueba
    
    Args:
        model: Modelo entrenado
        testloader: DataLoader para prueba
        
    Returns:
        tuple: (test_accuracy, class_correct, class_total)
    """
    
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(NUM_CLASSES))
    class_total = list(0. for i in range(NUM_CLASSES))
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Accuracy por clase
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Accuracy general
    test_accuracy = 100 * correct / total
    print(f"📈 Resultados en CIFAR-10:")
    print(f"   ✅ Test accuracy: {test_accuracy:.2f}%")
    print(f"   ✅ Correctas: {correct}/{total}")
    
    # Accuracy por clase
    print(f"📊 Accuracy por clase:")
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"   {i:2d}. {classes[i]:10s}: {class_accuracy:.2f}%")
    
    return test_accuracy, class_correct, class_total

# Evaluar modelo
test_accuracy, class_correct, class_total = evaluate_model(model, testloader)

# =============================================================================
# 8. CARGAR IMÁGENES PERSONALIZADAS DESDE CARPETA LOCAL
# =============================================================================
print("\n📂 Cargando imágenes personalizadas desde carpeta local...")

def load_custom_images_from_local():
    """
    Cargar imágenes personalizadas desde carpeta local
    
    Returns:
        tuple: (images, labels, filenames)
    """
    
    # Rutas locales donde buscar imágenes
    local_paths = [
        "./imagenes_prueba",
        "./100_imagenes_prueba", 
        "./Clasificador_Autos_Imagenes",
        "./imagenes_drive"
    ]
    
    images = []
    labels = []
    filenames = []
    
    # Transformaciones para las imágenes personalizadas
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Buscar carpeta con imágenes
    local_path = None
    for path in local_paths:
        if os.path.exists(path):
            local_path = path
            break
    
    if local_path is None:
        print("❌ No se encontró carpeta de imágenes local")
        print("💡 Asegúrate de que las imágenes estén en una de estas carpetas:")
        for path in local_paths:
            print(f"   - {path}")
        print("\n💡 Alternativa: Descarga las imágenes usando el script 'descargar_imagenes_prueba.py'")
        return [], [], []
    
    print(f"📂 Usando carpeta local: {local_path}")
    
    # Cargar imágenes por clase
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(local_path, class_name.lower())
        
        if not os.path.exists(class_dir):
            print(f"   ⚠️ Directorio no encontrado: {class_dir}")
            continue
        
        # Obtener todas las imágenes en el directorio
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"   📂 {class_name}: {len(image_files)} imágenes")
        
        for filename in image_files:
            try:
                filepath = os.path.join(class_dir, filename)
                
                # Cargar imagen
                img = Image.open(filepath).convert('RGB')
                
                # Preprocesar
                img_tensor = transform(img)
                
                # Guardar datos
                images.append(img_tensor)
                labels.append(class_idx)
                filenames.append(f"{class_name}/{filename}")
                
            except Exception as e:
                print(f"   ❌ Error procesando {filename}: {e}")
    
    print(f"✅ Total de imágenes cargadas: {len(images)}")
    return images, labels, filenames

# Cargar imágenes personalizadas
custom_images, custom_labels, custom_filenames = load_custom_images_from_local()

# =============================================================================
# 9. EVALUAR MODELO EN IMÁGENES PERSONALIZADAS
# =============================================================================
if len(custom_images) > 0:
    print("\n🔮 Evaluando modelo en imágenes personalizadas...")
    
    def evaluate_custom_images(model, images, labels, filenames):
        """
        Evaluar el modelo en imágenes personalizadas
        
        Args:
            model: Modelo entrenado
            images: Lista de tensores de imágenes
            labels: Lista de etiquetas reales
            filenames: Lista de nombres de archivo
            
        Returns:
            tuple: (predictions, confidences, results)
        """
        
        model.eval()
        predictions = []
        confidences = []
        results = []
        
        with torch.no_grad():
            for i, (img_tensor, true_label, filename) in enumerate(zip(images, labels, filenames)):
                # Hacer predicción
                img_tensor = img_tensor.unsqueeze(0).to(device)
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predictions.append(predicted.item())
                confidences.append(confidence.item())
                
                # Guardar resultado
                result = {
                    'filename': filename,
                    'true_class': classes[true_label],
                    'predicted_class': classes[predicted.item()],
                    'confidence': confidence.item(),
                    'correct': predicted.item() == true_label
                }
                results.append(result)
        
        # Calcular métricas
        correct_predictions = sum(1 for pred, true in zip(predictions, labels) if pred == true)
        accuracy = 100 * correct_predictions / len(labels)
        avg_confidence = np.mean(confidences) * 100
        
        print(f"📊 Resultados en imágenes personalizadas:")
        print(f"   ✅ Accuracy: {accuracy:.2f}%")
        print(f"   ✅ Correctas: {correct_predictions}/{len(labels)}")
        print(f"   ✅ Confianza promedio: {avg_confidence:.2f}%")
        
        return predictions, confidences, results
    
    # Evaluar imágenes personalizadas
    custom_preds, custom_confs, custom_results = evaluate_custom_images(
        model, custom_images, custom_labels, custom_filenames
    )
    
    # Análisis por clase en imágenes personalizadas
    print(f"\n📊 Análisis por clase (imágenes personalizadas):")
    for class_idx, class_name in enumerate(classes):
        class_indices = [i for i, label in enumerate(custom_labels) if label == class_idx]
        
        if class_indices:
            class_correct = sum(1 for i in class_indices if custom_preds[i] == class_idx)
            class_accuracy = 100 * class_correct / len(class_indices)
            print(f"   📂 {class_name}: {class_accuracy:.1f}% ({class_correct}/{len(class_indices)})")

# =============================================================================
# 10. VISUALIZAR RESULTADOS
# =============================================================================
print("\n📈 Visualizando resultados...")

def visualize_results():
    """Crear visualizaciones de los resultados"""
    
    plt.figure(figsize=(20, 12))
    
    # Subplot 1: Accuracy durante entrenamiento
    plt.subplot(2, 3, 1)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'b-', linewidth=2)
    plt.title('Accuracy durante Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    
    # Subplot 2: Accuracy por clase (CIFAR-10)
    plt.subplot(2, 3, 2)
    # Usar las variables globales correctas
    global class_correct, class_total
    class_accuracies = []
    for i in range(NUM_CLASSES):
        if class_total[i] > 0:
            class_accuracies.append(100 * class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0)
    bars = plt.bar(range(NUM_CLASSES), class_accuracies, color='orange', alpha=0.7)
    plt.title('Accuracy por Clase - CIFAR-10', fontsize=14, fontweight='bold')
    plt.xlabel('Clase')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(NUM_CLASSES), [name[:5] for name in classes], rotation=45)
    plt.ylim(0, 100)
    
    # Agregar valores en las barras
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Subplot 3: Comparación de accuracy
    plt.subplot(2, 3, 3)
    comparison_data = [test_accuracy]
    comparison_labels = ['CIFAR-10']
    
    if len(custom_images) > 0:
        custom_accuracy = 100 * sum(1 for pred, true in zip(custom_preds, custom_labels) if pred == true) / len(custom_labels)
        comparison_data.append(custom_accuracy)
        comparison_labels.append('Personalizadas')
    
    bars = plt.bar(comparison_labels, comparison_data, color=['blue', 'green'], alpha=0.7)
    plt.title('Comparación de Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Agregar valores en las barras
    for bar, acc in zip(bars, comparison_data):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # Subplot 4-6: Ejemplos de predicciones
    if len(custom_images) > 0:
        # Mostrar algunas imágenes con sus predicciones
        sample_indices = np.random.choice(len(custom_images), min(6, len(custom_images)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(2, 3, i+4)
            
            # Desnormalizar imagen para visualización
            img_tensor = custom_images[idx]
            img_display = img_tensor / 2 + 0.5
            img_display = img_display.permute(1, 2, 0).cpu().numpy()
            
            plt.imshow(img_display)
            
            result = custom_results[idx]
            title = f'Real: {result["true_class"]}\nPred: {result["predicted_class"]}\nConf: {result["confidence"]:.2f}'
            plt.title(title, fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Visualizaciones completadas")

visualize_results()

# =============================================================================
# 11. GUARDAR MODELO Y RESULTADOS
# =============================================================================
print("\n💾 Guardando modelo y resultados...")

# Guardar modelo
model_path = "modelo_autos_cnn_pytorch_local.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Modelo guardado localmente: {model_path}")

# Guardar resultados
if len(custom_images) > 0:
    # Crear DataFrame con resultados
    df_results = pd.DataFrame(custom_results)
    results_path = "resultados_predicciones_local.csv"
    df_results.to_csv(results_path, index=False)
    print(f"✅ Resultados guardados localmente: {results_path}")
    
    # Mostrar algunas predicciones
    print(f"\n📊 Ejemplos de predicciones:")
    for i, result in enumerate(custom_results[:5]):
        print(f"   {i+1}. {result['filename']}")
        print(f"      Real: {result['true_class']}")
        print(f"      Pred: {result['predicted_class']} (conf: {result['confidence']:.2f})")
        print()

# =============================================================================
# 12. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("🎉 ENTRENAMIENTO Y EVALUACIÓN COMPLETADOS")
print("=" * 70)

print(f"📊 RESULTADOS EN CIFAR-10:")
print(f"   ✅ Test accuracy: {test_accuracy:.2f}%")

if len(custom_images) > 0:
    custom_accuracy = 100 * sum(1 for pred, true in zip(custom_preds, custom_labels) if pred == true) / len(custom_labels)
    print(f"\n📊 RESULTADOS EN IMÁGENES PERSONALIZADAS:")
    print(f"   ✅ Accuracy: {custom_accuracy:.2f}%")
    print(f"   ✅ Total imágenes: {len(custom_images)}")

print(f"\n🏷️ CLASES CLASIFICADAS:")
for i, name in enumerate(classes):
    print(f"   {i}. {name}")

print(f"\n💾 ARCHIVOS GUARDADOS:")
print(f"   🧠 Modelo: {model_path}")
if len(custom_images) > 0:
    print(f"   📊 Resultados: {results_path}")

print(f"\n✅ ¡Proyecto completado exitosamente en ejecución local!")
print("🚀 Ventajas de esta implementación:")
print("   - ✅ Entrenamiento completo con CIFAR-10")
print("   - ✅ Evaluación con imágenes personalizadas")
print("   - ✅ Visualizaciones interactivas")
print("   - ✅ Guardado automático local")
print("   - ✅ Documentación completa")
print("   - ✅ Compatible con macOS")

# =============================================================================
# INSTRUCCIONES PARA EL USUARIO
# =============================================================================
print(f"\n📋 INSTRUCCIONES DE USO:")
print(f"1. 📂 Asegúrate de que las imágenes estén en una carpeta local")
print(f"2. 🧠 El modelo se guarda automáticamente en el directorio actual")
print(f"3. 📊 Los resultados se guardan en formato CSV")
print(f"4. 🔄 Puedes ejecutar este script múltiples veces")
print(f"5. 📈 Las visualizaciones se muestran automáticamente")

# =============================================================================
# PROTECCIÓN PARA MULTIPROCESSING EN MACOS
# =============================================================================
if __name__ == '__main__':
    # El código principal ya se ejecutó arriba
    pass
