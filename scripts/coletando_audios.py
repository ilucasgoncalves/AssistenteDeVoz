"""Script para coletar audios para treinar seu modelo de wake word 

    Para gravar ambiente, executar o som definido segundos a None. Isso vai
    gravar indefinidamente até ctrl + c

    Para gravar para um determinado período de tempo, defina segundos para o que quiser

    Para gravar interativamente (normalmente para gravando suas próprias palavras de despertar N vezes).
    use - modo --interativo
"""

import pyaudio
import wave
import argparse
import time
import os


class Escutar:

    def __init__(self, args):
        self.chunk = 1024
        self.FORMAT = pyaudio.paInt16
        self.canais = 1
        self.frequencia = args.frequencia
        self.gravar_segundos = args.segundos

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.FORMAT,
                        channels=self.canais,
                        rate=self.frequencia,
                        input=True,
                        output=True,
                        frames_per_buffer=self.chunk)


    def save_audio(self, file_name, frames):
        print('salvando arquivo em  {}'.format(file_name))
        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()

        # salvar audio
        wf = wave.open(file_name, "wb")
        # canais
        wf.setnchannels(self.canais)
        # formato
        wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
        # frequencia
        wf.setframerate(self.frequencia)
        wf.writeframes(b"".join(frames))
        # fechar arquivo
        wf.close()


def interactive(args):
    index = 0
    try:
        while True:
            escutar = Escutar(args)
            frames = []
            print('comecar a gravar....')
            input('aperte enter para continuar. a gravacao vai ser de {} segundos. aperte ctrl + c para sair '.format(args.segundos))
            time.sleep(0.2)  # para o microfone nao pegar barulhos de cliques
            for i in range(int((escutar.frequencia/escutar.chunk) * escutar.gravar_segundos)):
                data = escutar.stream.read(escutar.chunk, exception_on_overflow = False)
                frames.append(data)
            save_path = os.path.join(args.interativo_salvar, "{}.wav".format(index))
            escutar.save_audio(save_path, frames)
            index += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))


def main(args):
    escutar = Escutar(args)
    frames = []
    print('gravando...')
    try:
        while True:
            if escutar.gravar_segundos == None:  # gravar until keyboard interupt
                print('gravando indefinitivamente... ctrl + c para cancelar', end="\r")
                data = escutar.stream.read(escutar.chunk)
                frames.append(data)
            else:
                for i in range(int((escutar.frequencia/escutar.chunk) * escutar.gravar_segundos)):
                    data = escutar.stream.read(escutar.chunk)
                    frames.append(data)
                raise Exception('fim de gravacao')

    except KeyboardInterrupt:
        print('Keyboard Interrupt')
    except Exception as e:
        print(str(e))


    print('parando de gravar..')
    escutar.save_audio(args.save_path, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script para coletar dados para treinamento de wake word.

    Para gravar o ambiente, executar o som definido segundos a None. Isso vai
    gravar indefinidamente até ctrl + c.

    Para gravar para um determinado período de tempo, defina segundos ao que quiser.

    Para gravar interativamente (geralmente para gravando suas próprias palavras de despertar N vezes)
    use o modo --interativo.
    ''')
    parser.add_argument('--frequencia', type=int, default=8000,
                        help='a frequencia da gravacao em Hz')
    parser.add_argument('--segundos', type=int, default=None,
                        help='se definido para None, então gravará para sempre até a interrupção do teclado')
    parser.add_argument('--save_path', type=str, default=None, required=False,
                        help='caminho completo para salvar o arquivo. i.e. /to/path/sound.wav')
    parser.add_argument('--interativo_salvar', type=str, default=None, required=False,
                        help='diretório para salvar todas as amostras interativas de 2 segundos. i.e. /to/path/')
    parser.add_argument('--interativo', default=False, action='store_true', required=False,
                        help='modo interativo')

    args = parser.parse_args()

    if args.interativo:
        if args.interativo_salvar is None:
            raise Exception('precisa definir --interativo_salvar')
        interactive(args)
    else:
        if args.save_path is None:
            raise Exception('precisa definir --save_path')
        main(args)
