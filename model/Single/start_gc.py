import sys
sys.path.append('../..')
from base_gc import BASE_gc
import torch
from data.gc_dataloader import dataloader_gc
from torch_geometric.loader import DataLoader
from datetime import  datetime
import argparse

parser = argparse.ArgumentParser(description='BASE for Cross-Domain Graph Classification')
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--dataset_name', type=str, default='TUDataset', help='domain path')
parser.add_argument('--src_name', type=str, default='IMDB-BINARY', help='source domain')
parser.add_argument('--tgt_name', type=str, default='REDDIT-BINARY', help='target domain')
args = parser.parse_args()
args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")

src_dataset = dataloader_gc(model_name='pyg', dataset_name=args.dataset_name, collection_name=args.src_name)
tgt_dataset = dataloader_gc(model_name='pyg', dataset_name=args.dataset_name, collection_name=args.tgt_name)
src_loader = DataLoader(src_dataset, batch_size=64, shuffle=True)
tgt_loader = DataLoader(tgt_dataset, batch_size=64, shuffle=True)



def to_onehot(label_matrix, num_classes):
    identity = torch.eye(num_classes).to(args.device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot

def train():
    model.train()
    total_loss = 0.
    for idx, data in enumerate(src_loader):  # Iterate in batches over the training dataset.
         data = data.to(args.device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         y = to_onehot(data.y,num_classes)
         loss = criterion(out, y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
         total_loss += loss
    return total_loss/(idx+1)

def test(loader):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         data = data.to(args.device)
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

if 'BINARY' in args.src_name:
    input_feat = 136
    hidden_channels = 64
    num_classes = 2
elif 'Letter' in args.src_name:
    input_feat = 2
    hidden_channels = 32
    num_classes = 15


logfile = args.dataset_name.replace('/','-')+'-'+args.src_name+'-'+args.tgt_name+'.log'
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))
if __name__ == '__main__':
    avg_acc_list = []

    for r in range(3):
        best_acc = 0.

        model = BASE_gc(input_feat=input_feat,hidden_channels=hidden_channels,num_classes=num_classes).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, 300):
            loss = train()
            src_acc = test(src_loader)
            tgt_acc = test(tgt_loader)
            best_acc = max(best_acc, tgt_acc)
            print(f'Epoch: {epoch:03d}, loss:{loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}')
            with open(logfile, 'a+') as f:
                f.write(f'Epoch: {epoch:03d}, loss:{loss}, Source Acc: {src_acc:.4f}, Target Acc: {tgt_acc:.4f}, best acc:{best_acc}\n')
        avg_acc_list.append(best_acc)

    avg_acc = sum(avg_acc_list)/len(avg_acc_list)
    with open(logfile,'a+') as f:
        f.write(f"FINAL Avg acc:{avg_acc}\n")