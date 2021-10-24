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
For more details make sure to visit these files to look at script arguments and description

`wakeword/neuralnet/train.py` is used to train the model

`wakeword/neuralnet/optimize_graph.py` is used to create a production ready graph that can be used in `engine.py`

`wakeword/engine.py` is used to demo the wakeword model

`wakeword/scripts/collect_wakeword_audio.py` - used to collect wakeword and environment data

`wakeword/scripts/split_audio_into_chunks.py` - used to split audio into n second chunks

`wakeword/scripts/split_commonvoice.py` - if you download the common voice dataset, use this script to split it into n second chunks

`wakeword/scripts/create_wakeword_jsons.py` - used to create the wakeword json for training

### Steps to train and demo your wakeword model

For more details make sure to visit these files to look at script arguments and description

1. collect data
    1. environment and wakeword data can be collected using `python collect_wakeword_audio.py`
       ```
       cd VoiceAssistant/wakeword/scripts
       mkdir data
       cd data
       mkdir 0 1 wakewords
       python collect_wakeword_audio.py --sample_rate 8000 --seconds 2 --interactive --interactive_save_path ./data/wakewords
       ```
    2. to avoid the imbalanced dataset problem, we can duplicate the wakeword clips with 
       ```
       python replicate_audios.py --wakewords_dir data/wakewords/ --copy_destination data/1/ --copy_number 100
       ```
    3. be sure to collect other speech data like common voice. split the data into n seconds chunk with `split_audio_into_chunks.py`.
    4. put data into two seperate directory named `0` and `1`. `0` for non wakeword, `1` for wakeword. use `create_wakeword_jsons.py` to create train and test json
    5. create a train and test json in this format...
        ```
        // make each sample is on a seperate line
        {"key": "/path/to/audio/sample.wav, "label": 0}
        {"key": "/path/to/audio/sample.wav, "label": 1}
        ```

2. train model
    1. use `train.py` to train model
    2. after model training us `optimize_graph.py` to create an optimized pytorch model

3. test
    1. test using the `engine.py` script
