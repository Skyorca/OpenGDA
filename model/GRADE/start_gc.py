import sys
sys.path.append('../..')
from GRADE_gc import GRADE_GC
import torch
from data.dataloader import dataloader
from torch_geometric.loader import DataLoader
from datetime import  datetime
import argparse
from itertools import cycle
import os


parser = argparse.ArgumentParser(description='GRADE for Cross-network graph classification')
parser.add_argument('--dataset_name', type=str, default='TUDataset', help='domain path')
parser.add_argument('--src_name', type=str, default='IMDB-BINARY', help='source domain')
parser.add_argument('--tgt_name', type=str, default='REDDIT-BINARY', help='target domain')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg', type=float, default=0.0001, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--repeat', type=int, default=1,help='repear trainging times')
args = parser.parse_args()
args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")



if 'BINARY' in args.src_name:
    feat_dim = 136
    encoder_dim = 64
    num_classes = 2
    num_gcn_layers = 3
    hidden_dims = [feat_dim] + [encoder_dim]*num_gcn_layers

elif 'Letter' in args.src_name:
    feat_dim = 2
    encoder_dim = 32
    num_classes = 15
    num_gcn_layers = 3
    hidden_dims = [feat_dim] + [encoder_dim] * num_gcn_layers



def evaluate(loader):
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(args.device)
         out = model.inference(data)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


logfile = args.dataset_name.replace('/','-')+'-'+args.src_name+'-'+args.tgt_name+'.log'
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))

if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    src_dataset = dataloader(task_type="gc",model_name='pyg', dataset_name=args.dataset_name, collection_name=args.src_name)
    tgt_dataset = dataloader(task_type="gc",model_name='pyg', dataset_name=args.dataset_name, collection_name=args.tgt_name)
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=True)

    avg_acc_list = []

    for r in range(args.repeat):
        best_acc = 0.

        model =  GRADE_GC(in_feats=feat_dim, n_hidden=hidden_dims, n_classes=num_classes, n_layers=num_gcn_layers, dropout=args.dropout, device=args.device, disc='JS').to(device=args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs):
            model.train()
            total_loss = 0.
            total_domain_loss = 0.
            if len(src_loader)< len(tgt_loader):
                loader_pairwise = enumerate(zip(cycle(src_loader), tgt_loader))
            else:
                loader_pairwise = enumerate(zip(src_loader, cycle(tgt_loader)))
            for idx, (src_data, tgt_data) in loader_pairwise:  # Iterate in batches over the training dataset.
                src_data = src_data.to(args.device)
                tgt_data = tgt_data.to(args.device)
                loss, clf_loss, domain_loss = model(src_data, tgt_data)  # Perform a single forward pass.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                total_loss += loss
                total_domain_loss += domain_loss
            epoch_loss = total_loss/(idx+1)
            epoch_domain_loss = total_domain_loss/(idx+1)
            if epoch%1==0:
                model.eval()
                src_acc = evaluate(src_loader)
                tgt_acc = evaluate(tgt_loader)
                best_acc = max(best_acc, tgt_acc)
                print(f'Epoch: {epoch:03d}, loss:{epoch_loss}, domain_loss:{epoch_domain_loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}')
                with open(logfile, 'a+') as f:
                    f.write(f'Epoch: {epoch:03d}, loss:{epoch_loss}, domain_loss:{epoch_domain_loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}, best target acc:{best_acc}\n')
        avg_acc_list.append(best_acc)

    avg_acc = sum(avg_acc_list)/len(avg_acc_list)
    with open(logfile,'a+') as f:
        f.write(f"FINAL Avg acc:{avg_acc}\n")