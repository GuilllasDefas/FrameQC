"""
FrameQC - Módulo de Treinamento

Este script treina uma rede neural convolucional (CNN) para classificação binária
de imagens em 'Certos' e 'Errados'.
"""

import logging
# Configuração para suprimir avisos do TensorFlow
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Suprimir avisos do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar nível de log para reduzir mensagens
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

# Bibliotecas principais
import matplotlib.pyplot as plt
# TensorFlow e Keras
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # Suprimir mensagens de log do TensorFlow
"""
FrameQC - Módulo de Treinamento Aprimorado

Script otimizado para classificação binária de imagens usando Transfer Learning e técnicas avançadas.
"""

# Configuração de ambiente


import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import Input, layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuração de logs e warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')


def criar_data_generators(
    base_path, target_size=(224, 224), batch_size=32, val_split=0.2
):
    """
    Configura geradores de dados com aumento de dados otimizado para detecção de blur.

    Parâmetros:
        base_path (str): Caminho base para os dados
        target_size (tuple): Dimensões compatíveis com a rede pré-treinada
        batch_size (int): Tamanho ajustado para melhor utilização de memória
        val_split (float): Proporção para validação

    Retorna:
        tuple: (train_gen, val_gen), class_weights
    """
    print(f'\nConfigurando geradores de dados com tamanho {target_size}...')

    # Data augmentation otimizado para evitar falsos blur
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=2,
        brightness_range=[0.95, 1.05],
        zoom_range=0.01,
        width_shift_range=0.02,
        height_shift_range=0.02,
        fill_mode='constant',
        horizontal_flip=True,
        validation_split=val_split,
    )

    # Gerador de treino
    train_gen = train_datagen.flow_from_directory(
        base_path,
        target_size=target_size,
        batch_size=batch_size,
        classes=['Errados', 'Certos'],
        class_mode='binary',
        subset='training',
        shuffle=True,
    )

    # Gerador de validação (sem aumento de dados)
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=val_split
    )

    val_gen = val_datagen.flow_from_directory(
        base_path,
        target_size=target_size,
        batch_size=batch_size,
        classes=['Errados', 'Certos'],
        class_mode='binary',
        subset='validation',
        shuffle=False,
    )

    # Checagem: se não houver imagens, abortar com mensagem clara
    if train_gen.samples == 0 or val_gen.samples == 0:
        print("\nERRO: Nenhuma imagem encontrada no diretório fornecido.")
        print("Verifique se o diretório 'dataset' existe e contém subpastas 'Errados' e 'Certos' com imagens.")
        exit(1)

    # Cálculo de class weights para dados desbalanceados
    classes = train_gen.classes
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(classes), y=classes
    )
    class_weights = dict(enumerate(class_weights))

    return train_gen, val_gen, class_weights


def criar_modelo_avancado(input_shape=(224, 224, 3)):
    """
    Cria modelo baseado em Transfer Learning com MobileNetV2 e fine-tuning controlado.

    Parâmetros:
        input_shape (tuple): Dimensões de entrada compatíveis com a rede pré-treinada

    Retorna:
        Model: Modelo Keras compilado com métricas avançadas
    """
    print('\nConstruindo modelo com Transfer Learning...')

    # Carrega base model pré-treinada
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet'
    )

    # Congela parcialmente as camadas para fine-tuning
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Construção da arquitetura
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Camada base
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Compilação com otimizador personalizado
    model = models.Model(inputs, outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
        ],
    )

    model.summary()
    return model


def treinar_modelo(
    model, train_gen, val_gen, class_weights, epochs=50, models_dir=None
):
    """
    Rotina de treinamento com monitoramento aprimorado e callbacks.

    Parâmetros:
        model: Modelo compilado
        train_gen: Gerador de treino
        val_gen: Gerador de validação
        class_weights: Pesos para balanceamento de classes
        epochs: Número máximo de épocas
        models_dir: Diretório para salvar modelos

    Retorna:
        history: Histórico de treinamento
    """
    print('\nIniciando treinamento avançado...')

    # Callback para mostrar progresso detalhado durante cada época
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, num_epochs, report_interval=10):
            super().__init__()
            self.num_epochs = num_epochs
            self.report_interval = report_interval

        def on_epoch_begin(self, epoch, logs=None):
            print(f'\nÉpoca {epoch+1}/{self.num_epochs}')
            self.batch_count = 0
            self.start_time = time.time()

        def on_batch_end(self, batch, logs=None):
            self.batch_count += 1
            # Reportar a cada N batches para não sobrecarregar a saída
            if self.batch_count % self.report_interval == 0:
                elapsed = time.time() - self.start_time
                metrics = ' - '.join(
                    [f'{k}: {v:.4f}' for k, v in logs.items()]
                )
                print(
                    f'  Batch {self.batch_count} ({elapsed:.1f}s) - {metrics}'
                )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(models_dir, 'best_model.keras'),
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        ProgressCallback(
            epochs, report_interval=20
        ),  # Reportar a cada 20 batches
    ]

    # Importar time para medir duração
    import time

    # Calcular e mostrar estimativa de tempo total
    steps_per_epoch = len(train_gen)
    print(f'Total de {steps_per_epoch} batches por época')
    print(f'Máximo de {epochs} épocas')

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,  # Alterado para 1 para mostrar barra de progresso
    )

    return history


def visualizar_diagnosticos(history, train_gen, models_dir=None):
    """
    Gera visualizações completas do treinamento e exemplos de dados.

    Parâmetros:
        history: Histórico de treinamento
        train_gen: Gerador de treino para visualização de exemplos
        models_dir: Diretório para salvar gráficos
    """
    # Visualização de exemplos de treino
    print('\nVisualizando exemplos de dados aumentados...')
    images, labels = next(train_gen)
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f'Classe: {"Certo" if labels[i]>0.5 else "Errado"}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Gráficos de desempenho
    print('\nGerando métricas de treinamento...')
    plt.figure(figsize=(18, 6))

    # Acurácia e AUC
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.plot(history.history['auc'], label='AUC Treino')
    plt.plot(history.history['val_auc'], label='AUC Validação')
    plt.title('Desempenho do Modelo')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)

    # Precisão e Recall
    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision'], label='Precisão Treino')
    plt.plot(history.history['val_precision'], label='Precisão Validação')
    plt.plot(history.history['recall'], label='Recall Treino')
    plt.plot(history.history['val_recall'], label='Recall Validação')
    plt.title('Precisão e Recall')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)

    # Perda
    plt.subplot(1, 3, 3)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Função de Perda')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if models_dir:
        plt.savefig(os.path.join(models_dir, 'diagnostico_treinamento.png'))
    plt.show()


def main():
    """Fluxo principal de execução"""
    # Configurações
    base_path = 'dataset'
    target_size = (224, 224)
    batch_size = 64
    epochs = 100

    # Preparação de diretórios
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Pipeline de treinamento
    train_gen, val_gen, class_weights = criar_data_generators(
        base_path, target_size=target_size, batch_size=batch_size
    )

    model = criar_modelo_avancado(input_shape=target_size + (3,))

    history = treinar_modelo(
        model,
        train_gen,
        val_gen,
        class_weights,
        epochs=epochs,
        models_dir=models_dir,
    )

    visualizar_diagnosticos(history, train_gen, models_dir)

    # Salvar modelo final
    model.save(os.path.join(models_dir, 'modelo_final.keras'))
    print('\nTreinamento concluído e modelo salvo!')


if __name__ == '__main__':
    tf.random.set_seed(42)
    main()
