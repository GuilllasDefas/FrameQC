"""
FrameQC - Módulo de Treinamento

Este script treina uma rede neural convolucional (CNN) para classificação binária
de imagens em 'Certos' e 'Errados'.
"""

# Configuração para suprimir avisos do TensorFlow
import os
import warnings
import logging

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

from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def criar_data_generators(base_path, target_size=(128, 128), batch_size=16, val_split=0.2):
    """
    Configura geradores de dados para treino e validação com data augmentation.
    
    Parâmetros:
        base_path (str): Caminho base para os dados
        target_size (tuple): Dimensões para redimensionar as imagens
        batch_size (int): Tamanho do lote para processamento
        val_split (float): Proporção de dados para validação (0.0-1.0)
        
    Retorna:
        tuple: Geradores de treino e validação
    """
    print(f"Configurando geradores de dados a partir de: {base_path}")
    
    # Configuração do data augmentation
    # Aumenta a robustez do modelo, reduzindo overfitting
    datagen = ImageDataGenerator(
        # Normalização: essencial para convergência eficiente da rede
        rescale=1./255,                # Normaliza pixels de [0-255] para [0-1]
        
        # Data augmentation: gera variações artificiais dos dados de treino
        rotation_range=10,             # ↑ Valor = mais rotações aleatórias (+-20°)
        brightness_range=[0.8, 1.2],   # ↑ Range = mais variação de brilho
        zoom_range=0.05,               # ↑ Valor = mais variação de zoom (+-15%)
        width_shift_range=0.05,         # ↑ Valor = mais deslocamento horizontal
        height_shift_range=0.05,        # ↑ Valor = mais deslocamento vertical
        horizontal_flip=True,          # Espelhamento horizontal (útil se a orientação não importa)
        
        # Divisão treino/validação
        validation_split=val_split     # Reserva uma % das imagens para validação
    )
    
    # Para treino: usa (1-val_split)% das imagens
    print("Criando gerador para dados de TREINO...")
    train_gen = datagen.flow_from_directory(
        base_path,                     # Caminho contendo as subpastas 'Certos' e 'Errados'
        target_size=target_size,       # Redimensiona todas as imagens para o mesmo tamanho
        batch_size=batch_size,         # ↑ Batch = processamento mais rápido, ↓ Batch = mais estável
        classes=['Errados', 'Certos'], # Define explicitamente: Errados→0, Certos→1
        class_mode='binary',           # Saída binária (0 ou 1)
        subset='training'              # Usa a porção de treino definida pelo validation_split
    )
    
    # Para validação: usa val_split% das imagens
    print("Criando gerador para dados de VALIDAÇÃO...")
    val_gen = datagen.flow_from_directory(
        base_path,
        target_size=target_size,
        batch_size=batch_size,
        classes=['Errados', 'Certos'], # Mesma ordem para consistência
        class_mode='binary',
        subset='validation'            # Usa a porção de validação
    )
    
    return train_gen, val_gen


def criar_modelo(input_shape=(128, 128, 3)):
    """
    Cria e compila uma CNN para classificação binária com regularização.
    
    Parâmetros:
        input_shape (tuple): Forma dos dados de entrada (altura, largura, canais)
        
    Retorna:
        Model: Modelo Keras compilado
    """
    print(f"Criando modelo CNN com entrada de formato {input_shape}...")
    
    # Camada de entrada
    inputs = Input(shape=input_shape)
    
    # Bloco 1: Convolução + Pooling
    # Extração de características de baixo nível (bordas, texturas simples)
    x = layers.Conv2D(32, (3, 3), 
                     activation='relu',                          # ReLU: ativação não-linear padrão
                     kernel_regularizer=regularizers.l2(0.005)   # Regularização L2: ↑ valor = mais regularização
                     )(inputs)                                   # ↑ Regularização = menos overfitting, mas pode subajustar
    x = layers.MaxPooling2D(2, 2)(x)                             # Reduz dimensões espaciais pela metade
    
    # Bloco 2: Convolução + Pooling 
    # Extração de características de nível médio (formas, padrões)
    x = layers.Conv2D(64, (3, 3), 
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D(2, 2)(x)
    
    # Opcional: Bloco 3 para redes mais profundas (descomentado se precisar de mais capacidade)
    # x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    # x = layers.MaxPooling2D(2, 2)(x)
    
    # Flatten: transforma o mapa de características 2D em vetor 1D
    x = layers.Flatten()(x)
    
    # Camada densa: aprendizado de alto nível combinando características
    x = layers.Dense(64, activation='relu')(x)
    
    # Dropout: prevenção de overfitting desativando neurônios aleatoriamente durante treino
    x = layers.Dropout(0.3)(x)  # ↑ Taxa = mais regularização (0.3 = 30% de neurônios desativados)
    
    # Camada de saída: probabilidade da imagem ser "Certa"
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid fornece probabilidade [0-1]
    
    # Montagem do modelo
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compilação do modelo
    model.compile(
        optimizer='adam',              # Adam: otimizador adaptativo eficiente para maioria dos casos
        loss='binary_crossentropy',    # Função de perda para classificação binária
        metrics=['accuracy']           # Métrica primária: porcentagem de acertos
    )
    
    model.summary()  # Exibe arquitetura do modelo
    return model


def treinar_modelo(model, train_gen, val_gen, epochs=10):
    """
    Treina o modelo com os geradores de dados especificados.
    
    Parâmetros:
        model: Modelo Keras compilado
        train_gen: Gerador de dados de treino
        val_gen: Gerador de dados de validação
        epochs (int): Número máximo de épocas para treinamento
        
    Retorna:
        history: Histórico de treinamento
    """
    print(f"Iniciando treinamento por {epochs} épocas...")
    
    # Callbacks para melhor treinamento
    callbacks = [
        # Interrompe o treinamento quando a precisão de validação deixa de melhorar
        EarlyStopping(
            monitor='val_accuracy',    # Métrica monitorada
            patience=3,                # Número de épocas sem melhoria antes de parar
            restore_best_weights=True, # Restaura pesos do melhor checkpoint
            verbose=1                  # Mostra mensagens
        ),
        # Salva o melhor modelo durante o treinamento
        ModelCheckpoint(
            'best_model.keras',        # Atualizado para formato .keras recomendado
            monitor='val_accuracy',    # Métrica monitorada
            save_best_only=True,       # Salva apenas se for o melhor até agora
            verbose=1                  # Mostra mensagens
        )
    ]
    
    # Treina o modelo e captura o histórico
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def visualizar_resultados(history):
    """
    Cria gráficos para visualizar o histórico de treinamento.
    
    Parâmetros:
        history: Histórico retornado por model.fit()
    """
    print("Gerando visualizações dos resultados...")
    
    # Configuração de estilo
    plt.figure(figsize=(12, 5))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validação', marker='^')
    plt.title('Evolução da Acurácia')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Gráfico de perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino', marker='o')
    plt.plot(history.history['val_loss'], label='Validação', marker='^')
    plt.title('Evolução da Perda (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('historico_treinamento.png')  # Salva o gráfico
    plt.show()


def salvar_modelo(model, filename='FrameQC_model.keras'):
    """
    Salva o modelo treinado para uso posterior na subpasta 'models'.
    
    Parâmetros:
        model: Modelo Keras treinado
        filename (str): Nome do arquivo para salvar o modelo
    """
    # Determina o diretório do script atual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define o caminho para a subpasta 'models'
    models_dir = os.path.join(script_dir, 'models')
    
    # Cria a pasta 'models' se não existir
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Pasta 'models' criada em {models_dir}")
    
    # Caminho completo para salvar o modelo
    model_path = os.path.join(models_dir, filename)
    
    # Salva o modelo usando o formato .keras recomendado
    model.save(model_path)
    print(f'Modelo salvo em {model_path}')


def main():
    """Função principal para executar o fluxo de treinamento."""
    
    # 1. Define o caminho dos dados
    base_path = "dataset"  # Caminho base contendo subpastas 'Certos' e 'Errados'
    
    # 2. Cria geradores de dados
    train_gen, val_gen = criar_data_generators(
        base_path, 
        target_size=(128, 128),
        batch_size=16,
        val_split=0.1  # 20% para validação
    )
    
    # 3. Cria o modelo
    model = criar_modelo(input_shape=(128, 128, 3))
    
    # 4. Treina o modelo
    history = treinar_modelo(model, train_gen, val_gen, epochs=20)
    
    # 5. Visualiza os resultados
    visualizar_resultados(history)
    
    # 6. Salva o modelo
    salvar_modelo(model)


# Executa apenas quando o script é chamado diretamente
if __name__ == "__main__":
    # Configuração para reprodutibilidade
    tf.random.set_seed(42)
    
    # Executa o fluxo principal
    main()
