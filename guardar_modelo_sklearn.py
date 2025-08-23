# -*- coding: utf-8 -*-
"""
Script para guardar el modelo de scikit-learn entrenado
Autor: David Timana | Curso: Visión por Computador
"""

import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

print("💾 GUARDANDO MODELO SCIKIT-LEARN")
print("=" * 40)

# Cargar y entrenar modelo MNIST
print("📥 Cargando dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target

print("🔄 Preprocesando datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

print("🧠 Entrenando modelo...")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=50,  # Menos épocas para guardar rápido
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)

print("💾 Guardando modelo...")
joblib.dump(model, 'modelo_sklearn.pkl')
joblib.dump(scaler, 'scaler_sklearn.pkl')

print("✅ Modelo guardado como 'modelo_sklearn.pkl'")
print("✅ Scaler guardado como 'scaler_sklearn.pkl'")

# Evaluar modelo
y_pred = model.predict(scaler.transform(X_test))
accuracy = np.mean(y_pred == y_test) * 100
print(f"📊 Accuracy del modelo guardado: {accuracy:.2f}%")
