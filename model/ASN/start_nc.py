import sys
sys.path.append("..")
import argparse
import torch
import numpy as np
from ASN_nc import ASN_NC
from data.dataloader import dataloader
from datetime import datetime
from evaluation.metric import acc, multilabel_acc
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='ASN for Cross-network node classification')
parser.add_argument('--dataset_name', type=str, default='airport', help='domain path')
parser.add_argument('--src_name', type=str, default='usa', help='source domain')
parser.add_argument('--tgt_name', type=str, default='brazil', help='target domain')
parser.add_argument('--lr', type=float, default=3e-2, help='learning rate')
parser.add_argument('--reg', type=float, default=0.0001, help='weight decay')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epochs', type=int, default=300, help='epochs')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--diff',type=float, default=0.0001,help='loss coefficient')
parser.add_argument('--recon',type=float, default=0.0001,help='loss coefficient')   #  特别注意recon for airport
parser.add_argument('--domain',type=float, default=0.5,help='loss coefficient')
parser.add_argument('--repeat', type=int, default=1,help='repeat trainging times')
parser.add_argument('--which_model', type=str, help='flag in command to note which model you are running')

args = parser.parse_args()

if args.dataset_name=="airport":
    feat_dim = 8
    hidden_dim1 = 8
    hidden_dim2 = 8
    num_classes = 4
    num_gcn_layers = 2
    multi_label = 0
elif args.dataset_name=="citation":
    feat_dim = 6775
    hidden_dim1 = 128
    hidden_dim2 = 16
    num_classes = 5
    num_gcn_layers = 2
    multi_label = 1
elif args.dataset_name=="twitch":
    feat_dim = 3170
    hidden_dim1 = 64
    hidden_dim2 = 16
    num_classes = 2
    num_gcn_layers = 2
    multi_label = 0
elif args.dataset_name=="blog":
    feat_dim = 8189
    hidden_dim1 = 256
    hidden_dim2 = 64
    num_classes = 6
    num_gcn_layers = 2
    multi_label = 0


def evaluate(data,cache_name):
    logits = model.inference(data,cache_name)
    if multi_label:
        return multilabel_acc(y_true=data.y.cpu(), y_pred=logits.cpu())
    else:
        return acc(y_true=data.y.cpu(), y_pred=logits.cpu())


logfile = args.dataset_name.replace('/','-')+'-'+args.src_name+'-'+args.tgt_name+'.log'
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))

if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() and args.cuda>=0 else "cpu")
    src_dataset = dataloader(task_type="nc", model_name='pyg', dataset_name=args.dataset_name, collection_name=args.src_name)
    tgt_dataset = dataloader(task_type="nc", model_name='pyg', dataset_name=args.dataset_name, collection_name=args.tgt_name)
    src_data = src_dataset[0].to(args.device)
    tgt_data = tgt_dataset[0].to(args.device)
    avg_acc_list = []
    coeff = {"diff":args.diff, "recon":args.recon, "domain":args.domain}

    for r in range(args.repeat):
        best_acc = 0.

        model = ASN_NC(input_feat_dim=feat_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, num_classes=num_classes, dropout=args.dropout, coeff=coeff,device=args.device, multi_label=multi_label).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs):
            model.train()
            loss = model(src_data, tgt_data)  # Perform a single forward pass.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            if epoch%1==0:
                model.eval()
                src_acc = evaluate(src_data,"source")
                tgt_acc = evaluate(tgt_data,"target")
                best_acc = max(best_acc, tgt_acc)
                print(f'Epoch: {epoch:03d}, loss:{loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}')
                with open(logfile, 'a+') as f:
                    f.write(f'Epoch: {epoch:03d}, loss:{loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}, best acc:{best_acc}\n')
        avg_acc_list.append(best_acc)

    avg_acc = sum(avg_acc_list)/len(avg_acc_list)
    with open(logfile,'a+') as f:
        f.write(f"FINAL Avg acc:{avg_acc}\n")