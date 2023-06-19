import sys

sys.path.append("..")
import os
import torch.backends.cudnn as cudnn
from asn_base.gcn_encode import GCN, Attention, GCNModelVAE, InnerProductDecoder
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from utils import *
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class ASN_NC(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, num_classes, dropout, coeff, device, multi_label):
        super(ASN_NC, self).__init__()
        self.coeff = coeff  # 存储平衡loss各项的权重
        self.device = device
        self.multi_label = multi_label
        ''' private encoder/encoder for S/T (including Local GCN and Global GCN) '''
        self.private_encoder_s_l = GCNModelVAE(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2,
                                               dropout=dropout, device=self.device)
        self.private_encoder_t_l = GCNModelVAE(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2,
                                               dropout=dropout, device=self.device)
        self.private_encoder_s_g = GCNModelVAE(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2,
                                               dropout=dropout, type='ppmi', base_model=self.private_encoder_s_l,
                                               device=self.device)
        self.private_encoder_t_g = GCNModelVAE(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2,
                                               dropout=dropout, type='ppmi', base_model=self.private_encoder_t_l,
                                               device=self.device)
        self.decoder_s = InnerProductDecoder(dropout=dropout, act=lambda x: x)
        self.decoder_t = InnerProductDecoder(dropout=dropout, act=lambda x: x)

        ''' shared encoder (including Local GCN and Global GCN) '''
        self.shared_encoder_l = GCN(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2, dropout=dropout,
                                    device=self.device)
        self.shared_encoder_g = GCN(nfeat=input_feat_dim, nhid=hidden_dim1, nclass=hidden_dim2, dropout=dropout,
                                    type='ppmi', base_model=self.shared_encoder_l, device=self.device)

        ''' node classifier model '''
        self.cls_model = nn.Sequential(
            nn.Linear(hidden_dim2, num_classes)
        )

        # TODO: 判别器是否要设计得强一些
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
        ''' attention layer for local and global features '''
        self.att_model = Attention(hidden_dim2)
        self.att_model_self_s = Attention(hidden_dim2)
        self.att_model_self_t = Attention(hidden_dim2)

        self.criterion_d = nn.BCELoss()
        if self.multi_label:
            self.criterion_c = nn.BCEWithLogitsLoss()
        else:
            self.criterion_c = nn.CrossEntropyLoss()
        self.loss_diff = DiffLoss()

    def recon_loss(self, preds, labels, mu, logvar, n_nodes, norm, pos_weight):
        """
        :param preds: m*n的连边预测矩阵（二部图时m!=n）
        :param labels: m*n的实际连边矩阵（二部图时m!=n）
        :param mu:
        :param logvar:
        :param n_nodes: num_user+num_item
        :param norm:
        :param pos_weight:
        :return:
        """
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

    def forward(self, src_data, tgt_data,
                alpha=1.0):
        """
        这个模型几个点
        1. batch训练模式
        :param train_data_s:
        :param train_data_t:
        :param alpha:
        :return:
        """
        feats_s, adj_ds, batch_s, labels_s = src_data.x, src_data.edge_index, src_data.batch, src_data.y
        feats_t, adj_dt, batch_t = tgt_data.x, tgt_data.edge_index, tgt_data.batch
        n_node_s = feats_s.shape[0]
        n_node_t = feats_t.shape[0]

        # get the embeddings
        ## private
        recovered_s, mu_s, logvar_s = self.private_encoder_s_l(feats_s, adj_ds, "source")
        recovered_t, mu_t, logvar_t = self.private_encoder_t_l(feats_t, adj_dt, "target")

        recovered_s_p, mu_s_p, logvar_s_p = self.private_encoder_s_g(feats_s, adj_ds, "source")
        recovered_t_p, mu_t_p, logvar_t_p = self.private_encoder_t_g(feats_t, adj_dt, "target")
        ## share
        z_s, shared_encoded_source1, shared_encoded_source2 = self.shared_encoder_l(feats_s, adj_ds, "source")
        z_t, shared_encoded_target1, shared_encoded_target2 = self.shared_encoder_l(feats_t, adj_dt, "target")

        z_s_p, ppmi_encoded_source, ppmi_encoded_source2 = self.shared_encoder_g(feats_s, adj_ds, "source")
        z_t_p, ppmi_encoded_target, ppmi_encoded_target2 = self.shared_encoder_g(feats_t, adj_dt, "target")

        ''' the node representations after shared encoder for S and T '''
        # node classification loss
        x_ds = self.att_model([shared_encoded_source1, ppmi_encoded_source])
        x_dt = self.att_model([shared_encoded_target1, ppmi_encoded_target])
        source_logits = self.cls_model(x_ds)
        # TODO: 原来是label_s.float()
        if self.multi_label:
            clf_loss = self.criterion_c(source_logits, labels_s)
        else:
            clf_loss = self.criterion_c(source_logits,torch.argmax(labels_s, dim=1))
        # target entropy loss
        target_logits = self.cls_model(x_dt)
        target_probs = F.softmax(target_logits, dim=-1)
        target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
        entropy_loss = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
        # difference loss
        diff_loss_s = self.loss_diff(mu_s, shared_encoded_source1)
        diff_loss_t = self.loss_diff(mu_t, shared_encoded_target1)
        diff_loss = diff_loss_s + diff_loss_t
        ''' compute decoder reconstruction loss for S and T '''
        z_cat_s = torch.cat(
            (self.att_model_self_s([recovered_s, recovered_s_p]), self.att_model_self_s([z_s, z_s_p])), 1)
        z_cat_t = torch.cat(
            (self.att_model_self_t([recovered_t, recovered_t_p]), self.att_model_self_t([z_t, z_t_p])), 1)
        # difference in here
        recovered_cat_s = self.decoder_s(z_cat_s)
        recovered_cat_t = self.decoder_t(z_cat_t)
        mu_cat_s = torch.cat((mu_s, mu_s_p, shared_encoded_source1, ppmi_encoded_source), 1)
        mu_cat_t = torch.cat((mu_t, mu_t_p, shared_encoded_target1, ppmi_encoded_target), 1)
        logvar_cat_s = torch.cat((logvar_s, logvar_s_p, shared_encoded_source2, ppmi_encoded_source2), 1)
        logvar_cat_t = torch.cat((logvar_t, logvar_t_p, shared_encoded_target2, ppmi_encoded_target2), 1)
        # difference in here num_user_ds
        neg_adj_ds, adj_label_s, norm_s, pos_weight_s = get_loss_inputs(adj_ds, n_node_s,
                                                                        self.device)
        neg_adj_dt, adj_label_t, norm_t, pos_weight_t = get_loss_inputs(adj_dt, n_node_t,
                                                                        self.device)
        recover_s_pos = recovered_cat_s[adj_ds[0, :], adj_ds[1, :]]
        recover_s_neg = recovered_cat_s[neg_adj_ds[0, :], neg_adj_ds[1, :]]
        recover_t_pos = recovered_cat_t[adj_dt[0, :], adj_dt[1, :]]
        recover_t_neg = recovered_cat_t[neg_adj_dt[0, :], neg_adj_dt[1, :]]
        recover_s_pos_neg = torch.cat((recover_s_pos.view(1, -1), recover_s_neg.view(1, -1)), dim=1)[0]
        recover_t_pos_neg = torch.cat((recover_t_pos.view(1, -1), recover_t_neg.view(1, -1)), dim=1)[0]
        recon_loss_s = self.recon_loss(preds=recover_s_pos_neg, labels=adj_label_s,
                                       mu=mu_cat_s, logvar=logvar_cat_s, n_nodes=feats_s.shape[0],
                                       norm=norm_s, pos_weight=pos_weight_s)
        # TODO: WHY *2
        recon_loss_t = self.recon_loss(preds=recover_t_pos_neg, labels=adj_label_t,
                                       mu=mu_cat_t, logvar=logvar_cat_t, n_nodes=feats_t.shape[0] * 2,
                                       norm=norm_t, pos_weight=pos_weight_t)
        recon_loss = recon_loss_s + recon_loss_t
        # domain loss
        source_domain_preds = self.discriminator(ReverseLayerF.apply(x_ds, alpha))
        target_domain_preds = self.discriminator(ReverseLayerF.apply(x_dt, alpha))
        source_domain_labels = np.array([0] * source_domain_preds.shape[0])
        source_domain_labels = torch.tensor(source_domain_labels, requires_grad=False, dtype=torch.float).to(
            self.device)
        target_domain_labels = np.array([1] * target_domain_preds.shape[0])
        target_domain_labels = torch.tensor(target_domain_labels, requires_grad=False, dtype=torch.float).to(
            self.device)
        domain_loss = self.criterion_d(source_domain_preds.view(-1), source_domain_labels) + self.criterion_d(
            target_domain_preds.view(-1), target_domain_labels)

        print(f'clf:{clf_loss},diff:{diff_loss},recon:{recon_loss},domain:{domain_loss},entropy:{entropy_loss}')
        return clf_loss + self.coeff['diff'] * diff_loss + self.coeff['recon'] * recon_loss + self.coeff[
            'domain'] * domain_loss

    def inference(self, data, cache_name):
        assert cache_name in ['source','target']
        feat, adj= data.x, data.edge_index
        _, basic_encoded_output, _ = self.shared_encoder_l(feat, adj, cache_name)
        _, ppmi_encoded_output, _ = self.shared_encoder_g(feat, adj, cache_name)
        x =self.att_model([basic_encoded_output, ppmi_encoded_output])
        logits = self.cls_model(x)
        if self.multi_label:
            return F.sigmoid(logits)
        else:
            return F.softmax(logits)
