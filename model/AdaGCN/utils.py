import torch



def gradient_penalty(critic, src, tgt, device):
    Nodes_num = min(src.shape[0], tgt.shape[0])  # 梯度惩罚时的结点数
    features = src.shape[1]

    # create the interpolated nodes
    # alpha = torch.rand((Nodes_num, 1)).repeat(1,features).to(device)
    # random_points = alpha*(src[:Nodes_num]) + ((1 - alpha)*(tgt[:Nodes_num]))

    # # 下式用于将src,tgt,random_points拼接成新的数据
    # total_rep = torch.cat((src,tgt,random_points),0)

    # Calculate critic scores
    # mixed_scores = critic(random_points)
    inputs = torch.vstack([src, tgt])
    scores = critic(inputs)
    # # Take the gradient of the scores with respect to the image
    gradient = torch.autograd.grad(
        inputs=inputs,
        outputs=scores,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)  # L2 norm
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    # print(gradient_penalty)
    return gradient_penalty


def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot