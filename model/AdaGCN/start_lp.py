import sys
sys.path.append("../..")
import argparse
import torch
import numpy as np
from evaluation.metric import evaluate_test_at_k
from AdaGCN_lp import ADAGCN_LP
from data.dataloader import dataloader
from datetime import datetime
import time
from itertools import chain


parser = argparse.ArgumentParser(description='ADAGCN for Cross-Domain Recommendation')
parser.add_argument("--igcn", type=int,default=1)
parser.add_argument('--dataset_name', type=str, default='amazon1/nonoverlapping', help='domain path')
parser.add_argument('--src_name', type=str, default='music', help='source domain')
parser.add_argument('--tgt_name', type=str, default='cd', help='target domain')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--reg', type=float, default=0.0001, help='weight decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='epochs')
parser.add_argument('--lambda_b', type=int, default=1)
parser.add_argument('--lambda_gp', type=int, default=5)
parser.add_argument('--path_len', type=int, default=10, help='ppmi path len')
parser.add_argument('--cuda', type=int, default=0, help='cuda id')
parser.add_argument('--repeat', type=int, default=1,help='repear trainging times')
parser.add_argument('--which_model', type=str, help='flag in command to note which model you are running')

args = parser.parse_args()

if 'amazon1' in args.dataset_name:
    encoder_dim = 8
    num_gcn_layers = 2
    CRITIC_ITERATIONS = 10
    is_bipart_graph = 1  # 区分图是否为二部图，这会影响模型内部结构
    hidden_dims = [encoder_dim] * num_gcn_layers

elif 'citation' in args.dataset_name:
    encoder_dim = 16
    num_gcn_layers = 2
    CRITIC_ITERATIONS = 10
    is_bipart_graph = 0
    hidden_dims = [128, encoder_dim]


logfile = args.dataset_name.replace('/','-')+'-'+args.src_name+'-'+args.tgt_name+'.log'
with open(logfile, 'a+') as f:
    f.write("{0:%Y-%m-%d  %H-%M-%S/}\n".format(datetime.now()))

if __name__ == '__main__':
    coeff = {"LAMBDA":args.lambda_b,"LAMBDA_GP":args.lambda_gp}
    args.device = torch.device("cuda:{}".format(args.cuda) if (torch.cuda.is_available() and args.cuda>=0) else "cpu")
    src = dataloader('lp','pyg',args.dataset_name,args.src_name)[0].to(args.device)
    tgt = dataloader('lp','pyg',args.dataset_name,args.tgt_name)[0].to(args.device)
    test_dict_ds, test_input_dict_ds = src.test_dict, src.test_input_dict
    if is_bipart_graph:
        num_user_ds, num_item_ds = src.num_user,src.num_item
    else:
        num_user_ds, num_item_ds = src.num_node, src.num_node
    train_data_ds, adj_s, feats_s = src.data, src.edge_index, src.x
    test_dict_dt, test_input_dict_dt = tgt.test_dict, tgt.test_input_dict
    if is_bipart_graph:
        num_user_dt, num_item_dt = tgt.num_user,tgt.num_item
    else:
        num_user_dt, num_item_dt = tgt.num_node, tgt.num_node
    train_data_dt, adj_t, feats_t = tgt.data, tgt.edge_index, tgt.x

    hidden_dims = [feats_s.shape[1]] + hidden_dims

    avg_hits_list = []
    avg_mrr_list = []
    avg_ndcg_list = []

    for rep_times in range(args.repeat):
        model = ADAGCN_LP(args.igcn, encoder_dim=encoder_dim,num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,device=args.device, dropout=args.dropout,coeff=coeff, path_len=args.path_len, is_bipart_graph=is_bipart_graph).to(args.device)
        optimizer = torch.optim.Adam(params=chain(model.encoder.parameters(), model.cls_model.parameters()), lr=args.lr, weight_decay=args.reg)
        if is_bipart_graph:
            optimizer_critic = torch.optim.Adam(params=chain(model.discriminator_u.parameters(), model.discriminator_i.parameters()), lr=args.lr, weight_decay=args.reg)
        else:
            optimizer_critic = torch.optim.Adam(params=chain(model.discriminator.parameters()), lr=args.lr, weight_decay=args.reg)
        print(f"Repeat:{rep_times} Start training UDAGCN for cross-network link prediction on {args.dataset_name}, task:{args.src_name}->{args.tgt_name}")
        best_hits = 0.
        best_mrr = 0.
        best_ndcg = 0.
        for epoch in range(1, 1+args.epochs):
            model.train()
            permutation = torch.randperm(train_data_dt.shape[0])
            max_idx = int((len(permutation) // (args.batch_size / 2) - 1) * (args.batch_size / 2))

            loss = 0.
            for batch in range(0, max_idx, args.batch_size):
                optimizer.zero_grad()
                idx = permutation[batch: batch + args.batch_size]
                idx_s = np.random.choice(train_data_ds.shape[0], args.batch_size)
                # 采取的模式是train随机采样，test需要按照batch依次选取
                # train_data_s 和 train_data格式是一致的，train_data就是目标域，是测试集
                for inner_iter in range(CRITIC_ITERATIONS):
                    critic_loss = model.forward_critic(train_data_ds[idx_s, :], train_data_dt[idx], num_user_ds, num_user_dt, adj_s, adj_t, feats_s, feats_t)
                    optimizer_critic.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    optimizer_critic.step()
                    print('critic',critic_loss)
                loss = model(train_data_ds[idx_s, :], train_data_dt[idx], num_user_ds, num_user_dt, adj_s, adj_t, feats_s, feats_t)
                loss.backward()
                optimizer.step()

            if epoch % 1 == 0:
                model.eval()
                print("epoch {} loss: {:.4f}".format(epoch, loss))
                r = evaluate_test_at_k(model, test_dict_dt, test_input_dict_dt, k=10)
                best_hits = r['hits'] if r['hits']>best_hits else best_hits
                best_mrr = r['mrr'] if r['mrr']>best_mrr else best_mrr
                best_ndcg = r['ndcg'] if r['ndcg']>best_ndcg else best_ndcg

                with open(logfile,'a+') as f:
                    f.write(f"Epoch {epoch}: loss:{loss},hits:{r['hits']},mrr:{r['mrr']},ndcg:{r['ndcg']},best_hits:{best_hits},best_mrr:{best_mrr},best_ndcg:{best_ndcg}\n")
        print("Results:")
        model.eval()
        _ = evaluate_test_at_k(model, test_dict_dt, test_input_dict_dt, k=10)
        print(f"best Hits:{best_hits}, best MRR:{best_mrr}, best NDCG:{best_ndcg}")

        avg_hits_list.append(best_hits)
        avg_mrr_list.append(best_mrr)
        avg_ndcg_list.append(best_ndcg)

    avg_hits = sum(avg_hits_list)/args.repeat
    avg_mrr = sum(avg_mrr_list)/args.repeat
    avg_ndcg = sum(avg_ndcg_list)/args.repeat
    with open(logfile,'a+') as f:
        f.write(f"FINAL Avg hits:{avg_hits}, Avg mrr:{avg_mrr}, Avg ndcg:{avg_ndcg}\n")