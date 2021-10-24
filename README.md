# Criando um assistente pessoal usando Redes Neurais

Passo a passo de como criar o seu algoritimo de wakeword em Python. Mostramos como MFCC são usados para processamento de sinais de audio e como LSTMs podem ser usadas para reconhecimento de fala. https://youtu.be/4GokN_-4yh4

## Dependecias necessarias para rodar algoritimo esta em:
ambiente_virtual.yml

Esse algoritimo foi implementado e exectudado com as seguintes configurações:
Conda

Python 3.9.7

Pytorch 1.7.1

CUDA 11.1

GeForce RTX 3090

Ubuntu 18.04

### Crie um ambiente vitual chamado wakeword e ative:
1. conda env create -f ambiente_virtual.yml
2. conda activate wakeword

## Video passo a passo
[Youtube Video WakeWord]( https://youtu.be/4GokN_-4yh4)

### scripts
Descrição de como as pastas de scripts são organizadas:'

Siga instruções dentro de cada script para executar ou acessar o vídeo.

`AssistenteDeVoz/scripts/coletando_audios.py` usado para coletar audios

`AssistenteDeVoz/scripts/criando_json_para_treino.py ` usado para criar arquivo json de teste e treino da nossa rede neural

`AssistenteDeVoz/scripts/processando_audios.py` usado para processar audio criados para treino

`AssistenteDeVoz/scripts/processar_commonvoice.py ` - usado para processar audio obtidos de common voice 

`AssistenteDeVoz/scripts/replicando_audios.py ` - usado para replicar audio que criamos de wakeword

### rede_neural
Siga instruções dentro de cada script para executar ou acessar o vídeo.

`AssistenteDeVoz/rede_neural/treinar.py` usado para treinar nossa rede neural

`AssistenteDeVoz/rede_neural/executar.py  ` usado para testar nosso algoritimo de wakeword

`AssistenteDeVoz/rede_neural/modelo.py` model esta salvo aqui

`AssistenteDeVoz/rede_neural/modelo_optimizado.py ` - usado para criar modelo optimizado a partir de modelo treinado

`AssistenteDeVoz/rede_neural/dataset.py` usado para distribuir arquivos para nossa rede neural durante treinamento e teste

`AssistenteDeVoz/rede_neural/sonopy.py ` - necessario para processamento de audios

### Passos para treinar sua rede neural
Para mais detalhes acesse os arquivos e acesse o video

1. colete dados
    1. dados de ambiente e wakeword podem ser coletados usando `AssistenteDeVoz/scripts/coletando_audios.py`
       `` `
       cd AssistenteDeVoz/scripts/
       dados mkdir
       cd dados
       mkdir 0 1 wakewords
       python3 coletando_audios.py --frequencia 8000 --segundos 2 --interativo --interativo_salvar ./dados/wakewords
       `` `
    2. para evitar o problema do conjunto de dados desequilibrado, podemos duplicar os clipes de wakeword com
       `` `
       python3 replicando_audios.py --wakewords_dir dados/wakewords/ --copia_destino/1/ --numero_copias 300
       `` `
    3. certifique-se de coletar outros dados de fala, como voz comum. dividir os dados em pedaços de n segundos com `processando_audios.py`.
  
    5. Coloque os dados em dois diretórios separados chamados `0` e` 1`. `0` para não wakeword,` 1` para wakeword. use `criando_json_para_treino.py` para criar json de treino e teste neste formato ...
        `` `
        // fazer com que cada amostra esteja em uma linha separada
        {"key": "/path/to/audio/exemplo0.wav," label ": 0}
        {"key": "/path/to/audio/exemplo1.wav," label ": 1}
        `` `

2. treino o algoritimo de rede neural
    1. usar `treinar.py` para treinar o algoritimo
    2. após o treinamento do modelo, `modelo_optimizado.py` para criar um modelo pytorch otimizado

3. testando e executando modelo.
    1. teste usando o script `executar.py`
