import sys
sys.path.append("../..")
import argparse
import torch
import numpy as np
from AdaGCN_gc import ADAGCN_GC
from data.dataloader import dataloader
from datetime import datetime
import time
from torch_geometric.loader import DataLoader
from itertools import cycle, chain

parser = argparse.ArgumentParser(description='ADAGCN for Cross-network graph classification')
parser.add_argument("--igcn", type=int,default=0)
parser.add_argument('--dataset_name', type=str, default='TUDataset', help='domain path')
parser.add_argument('--src_name', type=str, default='IMDB-BINARY', help='source domain')
parser.add_argument('--tgt_name', type=str, default='REDDIT-BINARY', help='target domain')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg', type=float, default=0.0001, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--lambda_b', type=int, default=1)
parser.add_argument('--lambda_gp', type=int, default=5)
parser.add_argument('--path_len', type=int, default=5, help='ppmi path len')
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--repeat', type=int, default=1,help='repear trainging times')
parser.add_argument('--which_model', type=str, help='flag in command to note which model you are running')

args = parser.parse_args()

if 'BINARY' in args.src_name:
    feat_dim = 136
    encoder_dim = 64
    num_classes = 2
    num_gcn_layers = 3
    hidden_dims = [feat_dim] + [encoder_dim]*num_gcn_layers
    CRITIC_ITERATIONS = 10
elif 'Letter' in args.src_name:
    feat_dim = 2
    encoder_dim = 32
    num_classes = 15
    num_gcn_layers = 3
    hidden_dims = [feat_dim] + [encoder_dim]*num_gcn_layers
    CRITIC_ITERATIONS = 10

logfile = args.dataset_name.replace('/','-')+'-'+args.src_name+'-'+args.tgt_name+'.log'
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))


def evaluate(loader):
    print('begin evaluating')
    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(args.device)
        out = model.inference(data)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.

if __name__ == '__main__':
    coeff = {"LAMBDA":args.lambda_b,"LAMBDA_GP":args.lambda_gp}
    args.device = torch.device("cuda:{}".format(args.cuda) if (torch.cuda.is_available() and args.cuda>=0) else "cpu")
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    src_dataset = dataloader(task_type='gc',model_name='pyg', dataset_name=args.dataset_name, collection_name=args.src_name)
    tgt_dataset = dataloader(task_type='gc',model_name='pyg', dataset_name=args.dataset_name, collection_name=args.tgt_name)
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True)

    avg_acc_list = []

    for r in range(args.repeat):
        best_acc = 0.

        model = ADAGCN_GC(args.igcn, encoder_dim=encoder_dim,num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,num_classes=num_classes, device=args.device, dropout=args.dropout,coeff=coeff, path_len=args.path_len).to(args.device)
        optimizer = torch.optim.Adam(params=chain(model.encoder.parameters(), model.cls_model.parameters()), lr=args.lr, weight_decay=args.reg)
        optimizer_critic = torch.optim.Adam(params=chain(model.discriminator.parameters()), lr=args.lr,
                                            weight_decay=args.reg)
        for epoch in range(1, args.epochs):
            model.train()
            total_loss = 0.
            total_clf_loss = 0.
            total_d_loss = 0.
            if len(src_loader) < len(tgt_loader):
                loader_pairwise = enumerate(zip(cycle(src_loader), tgt_loader))
            else:
                loader_pairwise = enumerate(zip(src_loader, cycle(tgt_loader)))
            for idx, (src_data, tgt_data) in loader_pairwise:  # Iterate in batches over the training dataset.
                src_data = src_data.to(args.device)
                tgt_data = tgt_data.to(args.device)
                for inner_iter in range(CRITIC_ITERATIONS):
                    critic_loss = model.forward_critic(src_data, tgt_data)
                    optimizer_critic.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    optimizer_critic.step()
                    print('critic',critic_loss)

                loss,clf_loss,d_loss = model(src_data, tgt_data)  # Perform a single forward pass.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                total_loss += loss
                total_clf_loss += clf_loss
                total_d_loss += d_loss
            epoch_loss = total_loss / (idx + 1)
            epoch_clf_loss = total_clf_loss /(idx+1)
            epoch_d_loss = total_d_loss / (idx+1)
            if epoch % 1 == 0:
                model.eval()
                src_acc = evaluate(src_loader)
                tgt_acc = evaluate(tgt_loader)
                best_acc = max(best_acc, tgt_acc)
                print(f'Epoch: {epoch:03d}, loss:{epoch_loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}')
                with open(logfile, 'a+') as f:
                    f.write(
                        f'Epoch: {epoch:03d}, loss:{epoch_loss}, clf_loss:{epoch_clf_loss}, domain_loss；{epoch_d_loss}，Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}, best acc:{best_acc}\n')
        avg_acc_list.append(best_acc)

    avg_acc = sum(avg_acc_list) / len(avg_acc_list)
    with open(logfile, 'a+') as f:
        f.write(f"FINAL Avg acc:{avg_acc}\n")