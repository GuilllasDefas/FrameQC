# -*- coding: utf-8 -*-
"""
Script de Treino CNN Avançado para Classificação Binária
Autor: Sua Nome
Data: [Data]
"""

# =============================================
# Configuração Inicial e Supressão de Avisos
# =============================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprime avisos do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Otimizações para CPUs modernas

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Somente erros do TensorFlow

# Configurar precisão mista para melhor performance
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# =============================================
# Importações Necessárias
# =============================================
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import datetime

# =============================================
# Carregamento e Pré-processamento de Dados
# =============================================
def load_and_prepare_data():
    """Carrega e prepara os datasets de treino e validação"""
    # Carregar dados
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset',
        label_mode='binary',
        validation_split=0.2,
        subset='training',
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset',
        label_mode='binary',
        validation_split=0.2,
        subset='validation',
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )
    
    # Extrair nomes das classes antes das transformações
    class_names = train_ds.class_names
    
    # Otimização do pipeline de dados
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE).shuffle(1000)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    return train_ds, val_ds, class_names

train_dataset, val_dataset, class_names = load_and_prepare_data()
print(f"\nClasses detectadas: {class_names}")

# =============================================
# Data Augmentation
# =============================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
], name='data_augmentation')

# =============================================
# Construção do Modelo
# =============================================
def build_advanced_model():
    """Constrói o modelo com transfer learning e fine-tuning"""
    # Base do modelo
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar camadas para fine-tuning
    base_model.trainable = False  # Começa com todas congeladas
    
    # Descongelar camadas superiores
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    # Construção do modelo completo
    return models.Sequential([
        layers.Input(shape=(224, 224, 3), name='input_layer'),
        data_augmentation,
        layers.Rescaling(1./127.5, offset=-1, name='normalization'),
        base_model,
        layers.GlobalAveragePooling2D(name='gap_layer'),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', dtype='float32', name='dense_1'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid', dtype='float32', name='output_layer')
    ])

model = build_advanced_model()

# =============================================
# Configuração de Treinamento
# =============================================
def configure_training():
    """Configura otimizador, loss e métricas"""
    # Agendamento de learning rate
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=2*len(train_dataset),  # Ajusta a cada 2 épocas
        decay_rate=0.9,
        staircase=True
    )
    
    # Otimizador com weight decay
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        weight_decay=0.0001
    )
    
    # Função de perda e métricas
    loss = tf.keras.losses.BinaryCrossentropy()
    metrics = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc', curve='PR')  # AUC-ROC para dados desbalanceados
    ]
    
    return optimizer, loss, metrics

optimizer, loss, metrics = configure_training()
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# =============================================
# Callbacks e Class Weights
# =============================================
def get_class_weights(dataset):
    """Calcula pesos de classes balanceados"""
    labels = np.concatenate([y for x, y in dataset], axis=0)
    class_counts = np.bincount(labels.astype(int).flatten())
    total = class_counts.sum()
    return {
        0: total / (2 * class_counts[0]),
        1: total / (2 * class_counts[1])
    }

class_weights = get_class_weights(train_dataset)
print(f"\nClass weights: {class_weights}")

# Configurar callbacks
callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/best_model.keras',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    TensorBoard(
        log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )
]

# =============================================
# Treinamento do Modelo
# =============================================
print("\nIniciando treinamento...")
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# =============================================
# Visualização e Avaliação
# =============================================
def plot_training_metrics(history):
    """Plota métricas de treinamento"""
    plt.figure(figsize=(15, 6))
    
    # Perda
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Evolução da Perda')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()
    
    # AUC
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Treino')
    plt.plot(history.history['val_auc'], label='Validação')
    plt.title('Curva AUC')
    plt.ylabel('AUC')
    plt.xlabel('Época')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_metrics.png')
    plt.show()

plot_training_metrics(history)

# =============================================
# Avaliação Final e Matriz de Confusão
# =============================================
def evaluate_model(model, dataset):
    """Executa avaliação completa do modelo"""
    # Avaliação padrão
    results = model.evaluate(dataset, verbose=0)
    print("\nResultados da Avaliação:")
    print(f"- Perda: {results[0]:.4f}")
    print(f"- Acurácia: {results[1]:.4f}")
    print(f"- Precisão: {results[2]:.4f}")
    print(f"- Recall: {results[3]:.4f}")
    print(f"- AUC: {results[4]:.4f}")
    
    # Cálculo do F1-Score
    f1 = 2 * (results[2] * results[3]) / (results[2] + results[3])
    print(f"\nF1-Score: {f1:.4f}")
    
    # Matriz de Confusão
    y_true = np.concatenate([y for x, y in dataset], axis=0)
    y_pred = model.predict(dataset)
    y_pred = (y_pred > 0.5).astype(int)
    
    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm.numpy()[i][j], 
                    ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black')
    
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.savefig('models/confusion_matrix.png')
    plt.show()

evaluate_model(model, val_dataset)

# =============================================
# Salvamento Final
# =============================================
model.save('models/final_model.keras')
print("\nTreinamento concluído e modelo salvo com sucesso!")