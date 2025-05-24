# =============================================================================
# SISTEMA DE TREINAMENTO DE INTELIGÊNCIA ARTIFICIAL PARA CLASSIFICAÇÃO DE IMAGENS
# =============================================================================
# Este código treina uma IA para classificar imagens como "Verdadeiros" ou "Falsos"

import datetime  # Para criar nomes únicos de arquivos baseados na data/hora
import logging
# --- Configuração Inicial e Supressão de Avisos ---
# Estas configurações silenciam mensagens técnicas desnecessárias durante o treinamento
import os
import warnings

# Configurações para reduzir "poluição visual" no terminal
os.environ[
    'TF_CPP_MIN_LOG_LEVEL'
] = '2'  # 0=tudo, 1=info, 2=avisos, 3=só erros
os.environ[
    'TF_DETERMINISTIC_OPS'
] = '1'   # Faz a IA ser mais "previsível" nos resultados
warnings.filterwarnings(
    'ignore', category=UserWarning
)    # Ignora avisos gerais
warnings.filterwarnings(
    'ignore', category=FutureWarning
)  # Ignora avisos de futuras versões
logging.getLogger('tensorflow').setLevel(
    logging.ERROR
)    # Só mostra erros do TensorFlow
logging.getLogger('keras').setLevel(
    logging.ERROR
)         # Só mostra erros do Keras

# Importando as "ferramentas" que vamos usar
import tensorflow as tf  # Biblioteca principal para IA

tf.get_logger().setLevel(
    'ERROR'
)   # Mais uma configuração para silenciar mensagens

import itertools  # Para fazer loops especiais
import random  # Para gerar números aleatórios

import matplotlib.pyplot as plt  # Para criar gráficos bonitos
import numpy as np  # Para trabalhar com números e matrizes
from sklearn.metrics import auc  # Para medir o quão boa nossa IA ficou
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from tensorflow.keras import applications  # Peças para construir nossa IA
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (  # "Assistentes" durante o treinamento
    EarlyStopping, ModelCheckpoint, TensorBoard)

# --- Hiperparâmetros e Configurações Globais ---
# Pense nestes como "configurações do jogo" - mudando eles, mudamos como a IA aprende

# === CONFIGURAÇÕES DO DATASET (conjunto de imagens) ===
DATASET_DIR = 'dataset'  # Pasta onde estão nossas imagens de exemplo
IMAGE_SIZE = (
    64,
    64,
)  # Tamanho que todas as imagens terão (largura x altura em pixels)
# Maior = mais detalhes, mas mais lento para processar
BATCH_SIZE = 8         # Quantas imagens a IA vê de uma vez (como tamanho da "colherada") - # Batches menores para melhor precisão
# Menor = aprende mais devagar mas com mais precisão
VALIDATION_SPLIT = (
    0.2  # 20% das imagens serão usadas para "testar" a IA (não para treinar)
)
SEED = 123             # "Semente" para garantir que os resultados sejam reproduzíveis

# === CONFIGURAÇÕES DO TREINAMENTO ===
INITIAL_LR = 0.00005    # Taxa de aprendizado inicial - quão "grandes" são os ajustes que a IA faz
# Muito alto = aprende rápido mas pode "bagunçar", muito baixo = aprende devagar
DECAY_STEPS = 10000    # A cada quantos passos diminuímos a taxa de aprendizado
DECAY_RATE = (
    0.9  # Por quanto multiplicamos a taxa (0.9 = reduz 10% a cada vez)
)
EPOCHS = 50  # Quantas vezes a IA vai "estudar" todo o conjunto de imagens
PATIENCE_EARLY_STOPPING = (
    10  # Se a IA não melhorar por 7 épocas, paramos o treinamento
)
FINE_TUNE_AT_LAYER = 50     # A partir de qual camada vamos "ajustar finamente" a IA pré-treinada

# === CONFIGURAÇÕES DA ARQUITETURA (estrutura da IA) ===
DENSE_UNITS_1 = (
    512  # Quantidade de "neurônios" na primeira camada personalizada
)
DROPOUT_1 = (
    0.3  # Probabilidade de "desligar" neurônios (previne decorar demais)
)
DENSE_UNITS_2 = 256    # Quantidade de neurônios na segunda camada
DROPOUT_2 = 0.3        # Dropout da segunda camada

# --- Configuração de Seeds para Reprodutibilidade ---
# Isso garante que toda vez que rodarmos o código, teremos resultados similares
# É como fixar a "sorte" dos dados aleatórios
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- Preparação dos Diretórios ---
# Criamos pastas para salvar nossos modelos e logs (registros do treinamento)
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__ if '__file__' in globals() else '.')
)
MODELS_DIR = os.path.join(
    SCRIPT_DIR, 'models'
)  # Pasta para salvar a IA treinada
LOGS_DIR = os.path.join(
    SCRIPT_DIR,
    'logs',
    'fit',
    datetime.datetime.now().strftime('%Y%m%d-%H%M%S'),
)  # Logs com timestamp
os.makedirs(MODELS_DIR, exist_ok=True)   # Cria a pasta se não existir
os.makedirs(LOGS_DIR, exist_ok=True)     # Cria a pasta de logs

# --- Carregamento e Preparação do Dataset ---
# Aqui carregamos as imagens que vão "ensinar" nossa IA
print('\n--- Carregando Datasets ---')

# Carrega imagens para TREINAMENTO
# É como preparar um monte de flashcards para estudar
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,  # De onde pegar as imagens
    labels='inferred',  # As pastas determinam as classes (ex: pasta "Verdadeiros", pasta "Falsos")
    label_mode='binary',  # Classificação binária: apenas 2 categorias (0 ou 1)
    validation_split=VALIDATION_SPLIT,  # Separa 20% para validação
    subset='training',  # Esta é a parte de treino
    seed=SEED,  # Para reprodutibilidade
    image_size=IMAGE_SIZE,  # Redimensiona todas as imagens para o mesmo tamanho
    batch_size=BATCH_SIZE,  # Quantas imagens processar por vez
)

# Carrega imagens para VALIDAÇÃO
# São imagens que a IA nunca viu - usamos para testar se ela realmente aprendeu
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    labels='inferred',
    label_mode='binary',
    validation_split=VALIDATION_SPLIT,
    subset='validation',  # Esta é a parte de validação (teste)
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Descobrir quais são as classes (categorias) encontradas
CLASS_NAMES = train_ds.class_names
print(f'Classes encontradas: {CLASS_NAMES}')

# Verificação importante: garantir que as classes estão na ordem correta
# Para classificação binária, normalmente 0="Falso" e 1="Verdadeiro"
if CLASS_NAMES[0] != 'Falsos' or CLASS_NAMES[1] != 'Verdadeiros':
    print(f'Aviso: Ordem das classes detectada: {CLASS_NAMES}')
    print(
        "Esperado: ['Verdadeiros', 'Falsos'] para classificação binária correta"
    )

# Mostra quantas imagens temos para treinar e validar
print(
    f'Total de amostras de treino: {tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE}'
)
print(
    f'Total de amostras de validação: {tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE}'
)

# --- Data Augmentation (Aumento de Dados) ---
# Técnica para "multiplicar" nossas imagens criando variações
# É como tirar fotos do mesmo objeto de ângulos diferentes
# Isso ajuda a IA a não "decorar" as imagens específicas, mas aprender padrões gerais
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            'horizontal', seed=SEED
        ),  # Espelha a imagem horizontalmente (às vezes)
        layers.RandomRotation(
            0.05, seed=SEED
        ),  # Gira a imagem levemente (até 5%)
        layers.RandomZoom(0.15, seed=SEED),  # Zoom in/out aleatório (até 15%)
        #layers.RandomContrast(0.1, seed=SEED),  pode mascarar motion blur
    ],
    name='data_augmentation',
)

# --- Otimização de Performance do Dataset ---
# Configurações para que o carregamento das imagens seja mais rápido
AUTOTUNE = tf.data.AUTOTUNE  # Deixa o TensorFlow otimizar automaticamente
train_ds = train_ds.cache().prefetch(
    buffer_size=AUTOTUNE
)  # Cache + carregamento antecipado
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- Cálculo de Pesos de Classe (para lidar com desbalanceamento) ---
# Se tivermos muito mais imagens de uma classe que de outra, isso equilibra o aprendizado
print('\n--- Calculando Pesos de Classe ---')

# Primeiro, precisamos contar quantas imagens temos de cada classe
labels_list = []  # Lista para armazenar todos os rótulos (0s e 1s)

# Percorre todo o dataset de treino coletando os rótulos
for _, labels_batch in train_ds:  # Para cada lote de imagens e seus rótulos
    labels_list.append(labels_batch.numpy())  # Adiciona os rótulos à lista

# Junta todos os rótulos em uma única lista
labels_array = np.concatenate(labels_list, axis=0).astype(int).flatten()

# Se conseguimos coletar os rótulos, calculamos os pesos
if labels_array.size > 0:
    class_counts = np.bincount(labels_array)  # Conta quantos 0s e 1s temos

    if len(class_counts) == len(
        CLASS_NAMES
    ):  # Se temos contagens para todas as classes
        total_samples = np.sum(class_counts)   # Total de amostras

        # Fórmula para calcular pesos: classes com menos amostras ganham peso maior
        # Isso faz a IA dar mais atenção para a classe minoritária
        class_weights = {
            i: total_samples / (count * len(CLASS_NAMES))
            for i, count in enumerate(class_counts)
            if count > 0
        }

        print(
            f'Distribuição de classes no treino (antes do peso): {dict(enumerate(class_counts))}'
        )
        print(f'Pesos de classes aplicados: {class_weights}')
    else:
        print(
            'Aviso: Número de classes contadas não corresponde ao esperado. Não aplicando pesos de classe.'
        )
        class_weights = None
else:
    print(
        'Aviso: Não foi possível extrair labels para cálculo de pesos. Não aplicando pesos de classe.'
    )
    class_weights = None

# --- Construção do Modelo com Fine-Tuning ---
print('\n--- Construindo Modelo ---')

# Camada de normalização: converte imagens de 0-255 para -1 a 1
# As IAs pré-treinadas esperam este formato específico
rescale_layer = layers.Rescaling(
    1.0 / 255, offset=-1, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
)

# Modelo base: MobileNetV2 - uma IA já treinada em milhões de imagens
# É como contratar um professor que já sabe reconhecer formas básicas
base_model = applications.MobileNetV2(
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ),  # Formato das nossas imagens
    include_top=False,  # Remove a "cabeça" original - vamos colocar nossa própria
    weights='imagenet',  # Usa pesos de uma IA treinada no ImageNet (milhões de imagens)
)

# Inicialmente, "congelamos" o modelo base
# Significa que não vamos mudar o que ele já aprendeu (ainda)
base_model.trainable = False

# Fine-tuning: permite que as camadas finais do modelo base sejam ajustadas
# É como permitir que o professor adapte algumas técnicas para nosso problema específico
for layer in base_model.layers[FINE_TUNE_AT_LAYER:]:
    layer.trainable = True  # Permite treinar esta camada

print(
    f'Modelo base: {len(base_model.layers)} camadas. Fine-tuning a partir da camada {FINE_TUNE_AT_LAYER}.'
)
num_trainable_layers_base = sum(
    1 for layer in base_model.layers if layer.trainable
)
print(
    f'Número de camadas treináveis no modelo base: {num_trainable_layers_base}'
)


# Adicione após rescale_layer
grayscale_layer = layers.Lambda(
    lambda x: tf.image.rgb_to_grayscale(x),
    name='rgb_to_grayscale'
)

# Converte de volta para 3 canais (compatibilidade com MobileNetV2)
grayscale_to_rgb = layers.Lambda(
    lambda x: tf.concat([x, x, x], axis=-1),
    name='grayscale_to_rgb'
)

# Construção do modelo completo: modelo base + nossas camadas personalizadas
# É como empilhar blocos de Lego - cada camada processa a informação da anterior
model = models.Sequential([
    rescale_layer,           # 1. Normaliza as imagens  
    data_augmentation,       # 2. Aplica variações (antes do grayscale!)
    grayscale_layer,         # 3. Converte para escala de cinza
    grayscale_to_rgb,        # 4. Volta para 3 canais
    base_model,              # 5. Modelo MobileNetV2 pré-treinado
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(), 
    layers.Dense(DENSE_UNITS_1, activation='relu'),
    layers.Dropout(DROPOUT_1),
    layers.Dense(DENSE_UNITS_2, activation='relu'),
    layers.Dropout(DROPOUT_2),
    layers.Dense(1, activation='sigmoid'),
], name='Custom_MobileNetV2')

# --- Compilação do Modelo ---
# Aqui definimos como a IA vai aprender e quais métricas vamos acompanhar

# Learning Rate Schedule: diminui gradualmente a taxa de aprendizado
# É como começar estudando com passos grandes e depois com passos menores para precisão
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=INITIAL_LR,  # Taxa inicial
    decay_steps=DECAY_STEPS,  # A cada quantos passos diminui
    decay_rate=DECAY_RATE,  # Por quanto multiplica (0.9 = reduz 10%)
    staircase=True,  # Muda em intervalos discretos
)

# Otimizador: algoritmo que ajusta os "pesos" da IA para melhorar
# Adam é um dos melhores - funciona bem na maioria dos casos
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compilação: "monta" o modelo com todas as configurações
model.compile(
    optimizer=optimizer,  # Como vai aprender
    loss='binary_crossentropy',  # Como mede o "erro" (para 2 classes)
    metrics=[
        'accuracy',  # Métricas que queremos acompanhar:
        tf.keras.metrics.Precision(
            name='precision'
        ),  # Precisão: de todas que disse "sim", quantas estavam certas?
        tf.keras.metrics.Recall(name='recall'),
    ],  # Recall: de todas que eram "sim", quantas encontrou?
)

# Mostra a estrutura do modelo
model.summary()

# --- Callbacks ---
# "Assistentes" que monitoram o treinamento e tomam decisões automáticas
print('\n--- Configurando Callbacks ---')

# EarlyStopping: para o treinamento se não houver melhora
# É como um técnico que manda parar o treino se o atleta não está melhorando
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitora a "perda" na validação
    patience=PATIENCE_EARLY_STOPPING,  # Espera 7 épocas sem melhora
    verbose=1,  # Informa quando parar
    restore_best_weights=True,  # Volta para o melhor modelo encontrado
)

# ModelCheckpoint: salva automaticamente o melhor modelo (por acurácia)
# É como tirar uma foto sempre que o atleta quebra um recorde
model_checkpoint_accuracy = ModelCheckpoint(
    filepath=os.path.join(
        MODELS_DIR, 'best_model_val_accuracy.keras'
    ),  # Onde salvar
    save_best_only=True,  # Só salva se for melhor que o anterior
    monitor='val_accuracy',  # Monitora a acurácia de validação
    mode='max',  # Quer o valor MÁXIMO
    verbose=1,  # Informa quando salva
)

# Outro checkpoint para salvar o modelo com melhor precisão
model_checkpoint_f1 = ModelCheckpoint(
    filepath=os.path.join(MODELS_DIR, 'best_model_val_precision.keras'),
    save_best_only=True,
    monitor='val_precision',  # Monitora a precisão
    mode='max',
    verbose=1,
)

# TensorBoard: cria gráficos bonitos para visualizar o progresso
# É como um painel de controle que mostra tudo o que está acontecendo
tensorboard_callback = TensorBoard(
    log_dir=LOGS_DIR,  # Onde salvar os logs
    histogram_freq=1,  # Com que frequência salvar histogramas
    write_graph=True,  # Salva o gráfico da arquitetura do modelo
)

# Lista com todos os callbacks
callbacks_list = [
    early_stopping,
    model_checkpoint_accuracy,
    model_checkpoint_f1,
    tensorboard_callback,
]

# --- Treinamento do Modelo ---
# AQUI É ONDE A MÁGICA ACONTECE! A IA realmente aprende
print('\n--- Iniciando Treinamento ---')

# O método fit() é onde a IA "estuda" as imagens
history = model.fit(
    train_ds,  # Dados de treinamento (imagens + rótulos)
    validation_data=val_ds,  # Dados para testar durante o treino
    epochs=EPOCHS,  # Quantas vezes vai estudar todo o dataset
    callbacks=callbacks_list,  # Assistentes que monitoram o processo
    class_weight=class_weights
    if class_weights
    else None,  # Pesos para balancear classes
)

# --- Plotar Curvas de Treinamento ---
# Cria gráficos para visualizar como foi o aprendizado
print('\n--- Gerando Gráficos de Treinamento ---')

plt.figure(figsize=(14, 6))  # Cria uma figura com tamanho específico

# Primeiro gráfico: Loss (Perda)
# Mostra como o "erro" da IA diminuiu ao longo do tempo
plt.subplot(1, 2, 1)  # Primeiro gráfico de uma linha com 2 colunas
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda (Loss) Durante Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

# Segundo gráfico: Métricas (Acurácia, Precisão, Recall)
# Mostra como a performance da IA melhorou
plt.subplot(1, 2, 2)  # Segundo gráfico
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.plot(history.history.get('precision', []), label='Precisão Treino')
plt.plot(history.history.get('val_precision', []), label='Precisão Validação')
plt.plot(history.history.get('recall', []), label='Recall Treino')
plt.plot(history.history.get('val_recall', []), label='Recall Validação')
plt.title('Métricas Durante Treinamento')
plt.xlabel('Época')
plt.ylabel('Valor da Métrica')
plt.legend()

plt.tight_layout()  # Organiza os gráficos de forma bonita
plt.savefig(os.path.join(MODELS_DIR, 'training_history.png'))  # Salva a imagem
plt.show()  # Mostra os gráficos na tela

# --- Avaliação Final ---
# Agora vamos testar o quão boa nossa IA ficou
print(
    "\n--- Avaliação do Modelo com Melhor 'val_loss' (Restaurado por EarlyStopping) ---"
)

# Testa o modelo nas imagens de validação
val_loss, val_acc, val_prec, val_rec = model.evaluate(val_ds, verbose=1)

# Mostra os resultados de forma clara
print(f'Perda Validação: {val_loss:.4f}')         # Quanto menor, melhor
print(
    f'Acurácia Validação: {val_acc:.4f}'
)       # % de acertos (0-1, quanto maior melhor)
print(
    f'Precisão Validação: {val_prec:.4f}'
)      # Das que disse "sim", quantas eram realmente "sim"?
print(
    f'Recall Validação: {val_rec:.4f}'
)         # Das que eram "sim", quantas conseguiu encontrar?

# Calcula F1-Score: média harmônica entre Precisão e Recall
if (val_prec + val_rec) > 0:
    f1_score = 2 * (val_prec * val_rec) / (val_prec + val_rec)
    print(f'F1-Score Validação: {f1_score:.4f}')
else:
    print('F1-Score Validação: 0.0000 (divisão por zero evitada)')

# Salva o modelo treinado
model.save(os.path.join(MODELS_DIR, 'FrameQC_model.keras'))
print(
    f"Modelo com melhor 'val_loss' salvo em: {os.path.join(MODELS_DIR, 'FrameQC_model.keras')}"
)

# Avalia também o modelo que teve a melhor acurácia durante o treinamento
best_accuracy_model_path = os.path.join(
    MODELS_DIR, 'best_model_val_accuracy.keras'
)
if os.path.exists(best_accuracy_model_path):
    print(
        "\n--- Avaliação do Modelo com Melhor 'val_accuracy' (De ModelCheckpoint) ---"
    )

    # Carrega o modelo salvo automaticamente
    model_best_acc = tf.keras.models.load_model(best_accuracy_model_path)

    # Testa este modelo
    val_loss_ba, val_acc_ba, val_prec_ba, val_rec_ba = model_best_acc.evaluate(
        val_ds, verbose=1
    )

    # Mostra os resultados
    print(f'Perda Validação (Melhor Acurácia): {val_loss_ba:.4f}')
    print(f'Acurácia Validação (Melhor Acurácia): {val_acc_ba:.4f}')
    print(f'Precisão Validação (Melhor Acurácia): {val_prec_ba:.4f}')
    print(f'Recall Validação (Melhor Acurácia): {val_rec_ba:.4f}')

    if (val_prec_ba + val_rec_ba) > 0:
        f1_score_ba = (
            2 * (val_prec_ba * val_rec_ba) / (val_prec_ba + val_rec_ba)
        )
        print(f'F1-Score Validação (Melhor Acurácia): {f1_score_ba:.4f}')
    else:
        print(
            'F1-Score Validação (Melhor Acurácia): 0.0000 (divisão por zero evitada)'
        )

    # --- Matriz de Confusão e Relatório de Classificação ---
    # A matriz de confusão é como uma "tabela de acertos e erros"
    # Mostra exatamente onde nossa IA está errando
    print(
        '\n--- Gerando Matriz de Confusão para Modelo de Melhor Acurácia ---'
    )

    y_true_list = []        # Lista das respostas corretas
    y_pred_probs_list = []  # Lista das "apostas" da IA (probabilidades)

    # Para cada lote de imagens de validação
    for images_batch, labels_batch in val_ds:
        # Pede para a IA fazer predições
        predictions_batch = model_best_acc.predict_on_batch(images_batch)

        # Guarda as probabilidades que a IA calculou
        y_pred_probs_list.extend(predictions_batch.flatten())

        # Guarda as respostas corretas
        y_true_list.extend(labels_batch.numpy().astype(int).flatten())

    # --- Otimização de Threshold para Maximizar Precisão ---
    # Por padrão, se a IA diz >0.5, classificamos como "1" (Verdadeiro)
    # Mas podemos ajustar este limite para ter melhor precisão
    print('\n--- Otimizando Threshold para Máxima Precisão ---')

    # Calcula a curva ROC - mostra como a IA performa em diferentes thresholds
    fpr, tpr, thresholds = roc_curve(y_true_list, y_pred_probs_list)
    roc_auc = auc(fpr, tpr)  # AUC: área sob a curva (quanto maior, melhor)
    print(f'AUC: {roc_auc:.4f}')

    # Procura o melhor threshold (limite) para maximizar precisão
    from sklearn.metrics import precision_score

    best_threshold = 0.5    # Começamos com o padrão
    best_precision = 0      # Melhor precisão encontrada

    # Testa diferentes thresholds entre 0.3 e 0.8
    for threshold in np.arange(
        0.3, 0.8, 0.05
    ):  # De 0.3 a 0.8 em passos de 0.05
        # Converte probabilidades em decisões usando este threshold
        y_pred_thresh = [
            1 if prob > threshold else 0 for prob in y_pred_probs_list
        ]

        # Calcula a precisão com este threshold
        precision = precision_score(y_true_list, y_pred_thresh)

        # Se é melhor que o anterior, salva
        if precision > best_precision:
            best_precision = precision
            best_threshold = threshold

    print(
        f'Melhor threshold para precisão: {best_threshold:.2f} (Precisão: {best_precision:.4f})'
    )

    # Gera predições com ambos os thresholds para comparar
    y_pred_classes = [
        1 if prob > 0.5 else 0 for prob in y_pred_probs_list
    ]           # Threshold padrão
    y_pred_optimized = [
        1 if prob > best_threshold else 0 for prob in y_pred_probs_list
    ]  # Threshold otimizado

    # Função para criar gráficos da matriz de confusão
    def plot_confusion_matrix(
        cm, class_names_list, title='Matriz de Confusão', save_path=None
    ):
        """
        Cria um gráfico visual da matriz de confusão

        A matriz de confusão é uma tabela que mostra:
        - Linha: classe real (o que deveria ser)
        - Coluna: classe predita (o que a IA disse)
        - Diagonal principal: acertos
        - Fora da diagonal: erros
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(
            cm, interpolation='nearest', cmap=plt.cm.Blues
        )  # Cria o gráfico com cores azuis
        plt.title(title)
        plt.colorbar()  # Barra de cores para entender os valores

        # Configura os eixos com os nomes das classes
        tick_marks = np.arange(len(class_names_list))
        plt.xticks(
            tick_marks, class_names_list, rotation=45
        )  # Rótulos no eixo X
        plt.yticks(
            tick_marks, class_names_list
        )               # Rótulos no eixo Y

        # Adiciona os números dentro de cada célula da matriz
        thresh = cm.max() / 2.0  # Threshold para decidir cor do texto
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], 'd'),  # Número formatado
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black',
            )  # Cor do texto

        plt.tight_layout()
        plt.ylabel('Classe Real')      # Rótulo do eixo Y
        plt.xlabel('Classe Prevista')  # Rótulo do eixo X

        if save_path:
            plt.savefig(save_path)  # Salva a imagem se especificado
        plt.show()

    # Cria matriz de confusão com threshold padrão (0.5)
    cm = confusion_matrix(y_true_list, y_pred_classes)
    plot_confusion_matrix(
        cm,
        CLASS_NAMES,
        title='Matriz de Confusão (Threshold 0.5)',
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix_best_acc.png'),
    )

    # Relatório detalhado de classificação (threshold padrão)
    print('\nRelatório de Classificação (Threshold 0.5):')
    print(
        classification_report(
            y_true_list, y_pred_classes, target_names=CLASS_NAMES
        )
    )

    # Cria matriz de confusão com threshold otimizado
    cm_opt = confusion_matrix(y_true_list, y_pred_optimized)
    plot_confusion_matrix(
        cm_opt,
        CLASS_NAMES,
        title=f'Matriz de Confusão (Threshold {best_threshold:.2f})',
        save_path=os.path.join(MODELS_DIR, 'confusion_matrix_optimized.png'),
    )

    # Relatório detalhado de classificação (threshold otimizado)
    print(f'\nRelatório de Classificação (Threshold {best_threshold:.2f}:')
    print(
        classification_report(
            y_true_list, y_pred_optimized, target_names=CLASS_NAMES
        )
    )

else:
    print(
        f'Modelo {best_accuracy_model_path} não encontrado. Pulando avaliação e matriz de confusão.'
    )

print('\n--- Treinamento e Avaliação Concluídos ---')
print('🎉 Parabéns! Sua IA foi treinada com sucesso!')
print('📊 Verifique os gráficos e métricas para entender como ela performou.')
print("💾 Os modelos foram salvos na pasta 'models' para uso futuro.")
