"""Script de utilitário para criar arquivo json de treinamento para wakeword.

    Deve haver dois diretórios. aquele que tem todos os 0 rótulos
    e um com todos os 1 rótulos
"""
import os
import argparse
import json
import random


def main(args):
    zeros = os.listdir(args.zero_label_dir)
    ones = os.listdir(args.one_label_dir)
    percent = args.percent
    data = []
    for z in zeros:
        data.append({
            "key": os.path.join(args.zero_label_dir, z),
            "label": 0
        })
    for o in ones:
        data.append({
            "key": os.path.join(args.one_label_dir, o),
            "label": 1
        })
    random.shuffle(data)

    f = open(args.save_json_path +"/"+ "train.json", "w")
    
    with open(args.save_json_path +"/"+ 'train.json','w') as f:
        d = len(data)
        i=0
        while(i<int(d-d/percent)):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    
    f = open(args.save_json_path +"/"+ "test.json", "w")

    with open(args.save_json_path +"/"+ 'test.json','w') as f:
        d = len(data)
        i=int(d-d/percent)
        while(i<d):
            r=data[i]
            line = json.dumps(r)
            f.write(line + "\n")
            i = i+1
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""
    Script de utilitário para criar arquivo json de treinamento para wakeword.

    Deve haver dois diretórios. aquele que tem todos os 0 rótulos
    e um com todos os 1 rótulos
    """
    )
    parser.add_argument('--zero_label_dir', type=str, default=None, required=True,
                        help='diretório de clipes com 0 rótulos')
    parser.add_argument('--one_label_dir', type=str, default=None, required=True,
                        help='diretório de clipes com 1 rótulo')
    parser.add_argument('--save_json_path', type=str, default=None, required=True,
                        help='path to save json file')
    parser.add_argument('--percent', type=int, default=10, required=False,
                        help='porcentagem de clipes colocados em test.json em vez de train.json')
    args = parser.parse_args()

    main(args)
