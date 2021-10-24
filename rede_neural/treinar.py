"Script de treinamento"

import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from dataset import WakeWordData, collate_fn
from modelo import LSTMWW
from sklearn.metrics import classification_report
from tabulate import tabulate


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, model_params, notes=None):
    torch.save({
        "notes": notes,
        "model_params": model_params,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict()
    }, checkpoint_path)


def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    acc = rounded_preds.eq(y.view_as(rounded_preds)).sum().item() / len(y)
    return acc


def test(test_loader, model, device, epoch):
    print("\n começando o teste para epoch %s"%epoch)
    accs = []
    preds = []
    labels = []
    model.eval()
    with torch.no_grad():
        for idx, (mfcc, label) in enumerate(test_loader):
            mfcc, label = mfcc.to(device), label.to(device)
            output = model(mfcc)
            pred = torch.sigmoid(output)
            acc = binary_accuracy(pred, label)
            preds += torch.flatten(torch.round(pred)).cpu()
            labels += torch.flatten(label).cpu()
            accs.append(acc)
            print("Iter: {}/{}, Precisão: {}".format(idx, len(test_loader), acc), end="\r")
    average_acc = sum(accs)/len(accs) 
    print('Precisão média do teste:', average_acc, "\n")
    report = classification_report(labels, preds)
    print(report)
    return average_acc, report


def train(train_loader, model, optimizer, loss_fn, device, epoch):
    print("\n começando a treinar epoch %s"%epoch)
    losses = []
    preds = []
    labels = []
    model.train()
    for idx, (mfcc, label) in enumerate(train_loader):
        mfcc, label = mfcc.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(mfcc)
        pred = torch.round(torch.sigmoid(output))
        loss = loss_fn(torch.flatten(output), label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # obter previsões e rótulos para o relatório
        pred = torch.sigmoid(output)
        preds += torch.flatten(torch.round(pred)).cpu()
        labels += torch.flatten(label).cpu()

        print("epoch: {}, Iter: {}/{}, loss: {}".format(epoch, idx, len(train_loader), loss.item()), end="\r")
    avg_train_loss = sum(losses)/len(losses)
    acc = binary_accuracy(torch.Tensor(preds), torch.Tensor(labels))
    print('media de train loss:', avg_train_loss, "media de precisao de treino", acc)
    report = classification_report(torch.Tensor(labels).numpy(), torch.Tensor(preds).numpy())
    print(report)
    return acc, report


def main(args):
    use_cuda = True
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')

    train_dataset = WakeWordData(data_json=args.train_data_json, sample_rate=args.sample_rate, valid=False)
    test_dataset = WakeWordData(data_json=args.test_data_json, sample_rate=args.sample_rate, valid=True)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn,
                                        **kwargs)
    test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=args.eval_batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn,
                                        **kwargs)

    model_params = {
        "num_classes": 1, "feature_size": 40, "hidden_size": args.hidden_size,
        "num_layers": 1, "dropout" :0.1, "bidirectional": False
    }
    model = LSTMWW(**model_params, device=device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_train_acc, best_train_report = 0, None
    best_test_acc, best_test_report = 0, None
    melhor_epoch = 0
    for epoch in range(args.epochs):
        print("\ncomeçando o treinamento com taxa de aprendizagem", optimizer.param_groups[0]['lr'])
        train_acc, train_report = train(train_loader, model, optimizer, loss_fn, device, epoch)
        test_acc, test_report = test(test_loader, model, device, epoch)

        # registrar o melhor treinamento e teste
        if train_acc > best_train_acc:
            best_train_acc = train_acc
        if test_acc > best_test_acc:
            best_test_acc = test_acc

        # salva o ponto de verificação se as métricas forem melhores do que a anterior
        if args.save_checkpoint_path and test_acc >= best_test_acc:
            checkpoint_path = os.path.join(args.save_checkpoint_path, args.model_name + ".pt")
            print("encontrado o melhor checkpoint. salvar modelo como", checkpoint_path)
            save_checkpoint(
                checkpoint_path, model, optimizer, scheduler, model_params,
                notes="treino_precisao: {}, test_precisao: {}, epoch: {}".format(best_train_acc, best_test_acc, epoch),
            )
            best_train_report = train_report
            best_test_report = test_report
            melhor_epoch = epoch

        table = [["Treino Precisao", train_acc], ["Teste Precisao", test_acc],
                ["Melhor Precisao Treino", best_train_acc], ["Melhor Precisao Teste", best_test_acc],
                ["Melhor Epoch", melhor_epoch]]
        print(tabulate(table))

        scheduler.step(train_acc)

    print("Treinamento concluído ...")
    print("Melhor modelo salvo em", checkpoint_path)
    print("Melhor Epoch", melhor_epoch)
    print("\nTrain Report \n")
    print(best_train_report)
    print("\nTest Report\n")
    print(best_test_report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake Word Script de treinamento")
    parser.add_argument('--sample_rate', type=int, default=8000, help='sample_rate for data')
    parser.add_argument('--epochs', type=int, default=50, help='epoch size')
    parser.add_argument('--batch_size', type=int, default=64, help='tamanho do batch')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='tamanho do batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--model_name', type=str, default="wakeword", required=False, help='nome do modelo para salvar')
    parser.add_argument('--save_checkpoint_path', type=str, default=None, help='Caminho para salvar o melhor checkpoint')
    parser.add_argument('--train_data_json', type=str, default=None, required=True, help='caminho para treinar arquivo json de dados')
    parser.add_argument('--test_data_json', type=str, default=None, required=True, help='caminho  para testar o arquivo json de dados')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='desativa o treinamento CUDA')
    parser.add_argument('--num_workers', type=int, default=1, help='número de workers de carregamento de dados')
    parser.add_argument('--hidden_size', type=int, default=128, help='lstm hidden size')

    args = parser.parse_args()

    main(args)
