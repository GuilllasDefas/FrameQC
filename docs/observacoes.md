# Obervações

## Cores e Processamento

- As cores, neste contexto, não são relevantes, pois o foco é a figura central e sua nitidez.
- Conversão para grayscale reduz complexidade computacional em 66% (1 canal vs 3 canais RGB).
- Grayscale facilita detecção de motion blur, que é baseado em gradientes de intensidade.

## Tamanho da Imagem

- O tamanho da imagem influencia muito no reconhecimento de padrões neste caso, pois imagens pequenas dificultam a identificação de nitidez.
- O tamanho da imagem influencia muito no tempo de treinamento, pois imagens grandes exigem mais processamento.
- Recomendação: começar com 224x224 para detectar motion blur fino, testar 384x384 se necessário mais detalhes (diminuindo batches).
- Imagens menores que 224x224 podem não capturar detalhes sutis de postura e movimento.

## Data Augmentation

- O data augmentation é uma técnica utilizada para aumentar a quantidade de dados, mas deve se ter cuidado com o layers.RandomContrast, pois pode mascarar detalhes importantes da imagem.
- RandomZoom é crucial para simular aproximação/afastamento da câmera.
- RandomRotation deve ser limitado (≤0.05) para manter naturalidade dos movimentos corporais.
- RandomBrightness ajuda com variações de iluminação sem mascarar motion blur.
- Data augmentation deve ser aplicado ANTES da conversão para grayscale para melhores resultados.

## Arquitetura do Modelo

- Fine-tuning agressivo (camada 50 em vez de 100) permite melhor adaptação para detecção de detalhes específicos.
- Camadas densas maiores (512, 256) aumentam capacidade de detectar características complexas.
- GlobalAveragePooling2D preserva informações espaciais importantes para análise postural.

## Desbalanceamento de Classes

- Pesos de classe são essenciais quando há disparidade entre "Verdadeiros" e "Falsos".
- Monitorar distribuição do dataset para evitar overfitting em classe majoritária.

## Métricas e Validação

- Foco em alta Precisão para evitar falsos positivos (imagens ruins classificadas como boas).
- Recall também importante para não perder imagens realmente problemáticas.
- Validação cruzada recomendada devido à natureza específica do problema.

## Performance e Otimização

- Cache e prefetch essenciais para datasets grandes de imagens.
- Batch size menor (8-16) permite melhor precisão em detalhes finos.
- Learning rate baixo (0.00005) necessário para fine-tuning preciso.
- EficientNetB0 é muito pesado, mas pode ser necessário para detalhes finos. Por isso MobileNetV2 para comparação foi escolhido.

## Casos Específicos do Problema

- Motion blur pode ser localizado (ex: só na mão) - modelo deve detectar qualquer região com blur.
- Postura incorreta inclui: olhar para trás, olhar para baixo, movimentação excessiva.
- Pessoas ao redor são ruído - modelo deve focar apenas na figura central.
- Variação de distância da câmera é normal - zoom augmentation simula isso.
