import os
import argparse
import shutil

def main(args):
    ones = os.listdir(args.wakewords_dir)
    dest_dir = os.mkdir(args.wakewords_dir+'subfolder')
    os.listdir()
    for file in ones:
        if file.endswith(".wav") or file.endswith(".mp3"):
            for i in range(args.numero_copias):
                # copy
                dest_dir = args.copia_destino
                srcFile = os.path.join(args.wakewords_dir, file)
                shutil.copy(srcFile, dest_dir)
                # rename the file in the subfolder
                dst_file = os.path.join(dest_dir, file)
                new_dst_file = os.path.join(dest_dir, str(i) + "_" + file)
                os.rename(dst_file, new_dst_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Script de utilitário para replicar os clipes de wakeword várias vezes.

    """
    )
    parser.add_argument('--wakewords_dir', type=str, default=None, required=True,
                        help='diretório de clipes com wakewords')

    parser.add_argument('--copia_destino', type=str, default=None, required=True,
                        help='diretório dos destinos das cópias wakewords')

    parser.add_argument('--numero_copias', type=int, default=100, required=False,
                        help='o número de cópias que você deseja')

    args = parser.parse_args()

    main(args)