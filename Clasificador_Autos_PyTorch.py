# -*- coding: utf-8 -*-
"""
Clasificador de Autos CNN - Versión PyTorch
Basado en el estilo del profesor - Compatible con macOS
Autor: David Timana | Curso: Visión por Computador
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

print("🚗 CLASIFICADOR DE AUTOS CNN - VERSIÓN PYTORCH")
print("=" * 50)

# Verificar configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ Device: {device}")

# =============================================================================
# 1. CARGAR DATASET CIFAR-10
# =============================================================================
print("\n1. Cargando dataset CIFAR-10...")

# Transformaciones para preprocesamiento
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Cargar datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Crear dataloaders (sin multiprocessing para compatibilidad con macOS)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

# Clases disponibles
classes = ('avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 
           'perro', 'rana', 'caballo', 'barco', 'camión')

print(f"✅ Dataset cargado:")
print(f"   📊 Entrenamiento: {len(trainset):,} imágenes")
print(f"   📊 Prueba: {len(testset):,} imágenes")
print(f"   🖼️  Tamaño: 32x32x3")
print(f"   🏷️  Clases: 10")
print(f"   📝 Clases: {classes}")
print(f"   🚗 Automóviles: clase 1")

# =============================================================================
# 2. DEFINIR MODELO CNN
# =============================================================================
print("\n2. Construyendo modelo CNN...")

class CNN(nn.Module):
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
        
        # Capas densas
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Primera capa convolucional
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # Segunda capa convolucional
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # Tercera capa convolucional
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # Aplanar
        x = x.view(-1, 128 * 4 * 4)
        
        # Capas densas
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
# 3. DEFINIR FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# =============================================================================
print("\n3. Configurando entrenamiento...")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"✅ Función de pérdida: CrossEntropyLoss")
print(f"✅ Optimizador: Adam (lr=0.001)")

# =============================================================================
# 4. ENTRENAR MODELO
# =============================================================================
print("\n4. Entrenando modelo...")

def train_model(epochs=10):
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"🚀 Época {epoch+1}/{epochs}")
        
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
                print(f'   Batch {i+1}, Loss: {running_loss/100:.3f}, Accuracy: {100*correct/total:.2f}%')
                running_loss = 0.0
        
        # Calcular accuracy de la época
        epoch_accuracy = 100 * correct / total
        train_accuracies.append(epoch_accuracy)
        train_losses.append(running_loss / len(trainloader))
        
        print(f"✅ Época {epoch+1} completada - Accuracy: {epoch_accuracy:.2f}%")
    
    return train_losses, train_accuracies

# Entrenar modelo
print("🚀 Iniciando entrenamiento...")
train_losses, train_accuracies = train_model(epochs=10)
print("✅ Entrenamiento completado!")

# =============================================================================
# 5. EVALUAR MODELO
# =============================================================================
print("\n5. Evaluando modelo...")

def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
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
    print(f"📈 Resultados:")
    print(f"   ✅ Test accuracy: {test_accuracy:.2f}%")
    
    # Accuracy por clase
    print(f"📊 Accuracy por clase:")
    for i in range(10):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
            print(f"   {i:2d}. {classes[i]:10s}: {class_accuracy:.2f}%")
    
    return test_accuracy, class_correct, class_total

test_accuracy, class_correct, class_total = evaluate_model()

# =============================================================================
# 6. VISUALIZAR RESULTADOS
# =============================================================================
print("\n6. Visualizando resultados...")

# Gráfico de accuracy durante entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'b-', label='Train Accuracy')
plt.title('Accuracy durante Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Gráfico de accuracy por clase
plt.subplot(1, 2, 2)
class_accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(10)]
bars = plt.bar(range(10), class_accuracies, color='orange')
plt.title('Accuracy por Clase')
plt.xlabel('Clase')
plt.ylabel('Accuracy (%)')
plt.xticks(range(10), [f'{i}\n{name[:3]}' for i, name in enumerate(classes)], fontsize=8)

# Agregar valores en las barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

print("✅ Visualizaciones completadas!")

# =============================================================================
# 7. PREDICCIONES EN IMÁGENES ESPECÍFICAS
# =============================================================================
print("\n7. Predicciones en imágenes específicas...")

def show_predictions():
    model.eval()
    
    # Obtener algunas imágenes de prueba
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    
    # Mostrar primeras 5 imágenes
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        
        # Desnormalizar imagen
        img = images[i] / 2 + 0.5
        img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        
        # Hacer predicción
        with torch.no_grad():
            output = model(images[i:i+1].to(device))
            _, predicted = torch.max(output, 1)
            confidence = torch.softmax(output, dim=1).max().item()
        
        plt.title(f'Real: {classes[labels[i]]}\nPred: {classes[predicted[0]]}\nConf: {confidence:.2f}')
        plt.axis('off')
    
    plt.suptitle('Predicciones en Imágenes de Prueba', fontsize=16)
    plt.tight_layout()
    plt.show()

show_predictions()
print("✅ Predicciones completadas!")

# =============================================================================
# 8. ANÁLISIS DE CLASE ESPECÍFICA (AUTOMÓVILES)
# =============================================================================
print("\n8. Análisis específico de automóviles...")

def analyze_cars():
    model.eval()
    car_correct = 0
    car_total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            
            # Filtrar solo automóviles (clase 1)
            car_mask = (labels == 1)
            if car_mask.sum() > 0:
                car_images = images[car_mask]
                car_labels = labels[car_mask]
                
                outputs = model(car_images)
                _, predicted = torch.max(outputs, 1)
                
                car_total += car_labels.size(0)
                car_correct += (predicted == car_labels).sum().item()
    
    if car_total > 0:
        car_accuracy = 100 * car_correct / car_total
        print(f"🚗 Automóviles en conjunto de prueba: {car_total}")
        print(f"✅ Precisión en automóviles: {car_accuracy:.2f}%")
        return car_accuracy
    else:
        print("❌ No se encontraron automóviles en el conjunto de prueba")
        return 0

car_accuracy = analyze_cars()

# =============================================================================
# 9. GUARDAR MODELO
# =============================================================================
print("\n9. Guardando modelo...")

torch.save(model.state_dict(), 'modelo_autos_cnn_pytorch.pth')
print("✅ Modelo guardado como 'modelo_autos_cnn_pytorch.pth'")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 50)
print("🎉 ENTRENAMIENTO COMPLETADO")
print("=" * 50)
print(f"📊 Test accuracy: {test_accuracy:.2f}%")
print(f"🏷️  Clases: 10 (incluyendo automóviles)")
print(f"🧠 Modelo: CNN (PyTorch)")
print(f"💻 Device: {device}")
print(f"🚗 Precisión en automóviles: {car_accuracy:.2f}%")
print(f"💾 Modelo guardado: modelo_autos_cnn_pytorch.pth")

print("\n🎯 El modelo puede clasificar:")
for i, name in enumerate(classes):
    print(f"   {i}. {name}")

print("\n✅ ¡Proyecto completado exitosamente con PyTorch!")
print("🚀 Ventajas de PyTorch:")
print("   - ✅ Excelente compatibilidad con macOS")
print("   - ✅ Sintaxis más intuitiva")
print("   - ✅ Debugging más fácil")
print("   - ✅ Flexibilidad en el diseño de modelos")
print("   - ✅ Gran comunidad y documentación")

# =============================================================================
# PROTECCIÓN PARA MULTIPROCESSING EN MACOS
# =============================================================================
if __name__ == '__main__':
    # El código principal ya se ejecutó arriba
    pass
