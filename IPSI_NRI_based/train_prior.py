import torch.optim as optim
from modules_SI_prior import *
import argparse
from utils import *
import os
# 训练函数
def train_SIprior_model(emb_SI_prior, encoder_SI_prior,  train_loader, optimizer_SIprior, device,rel_rec, rel_send):
    emb_SI_prior.train()
    # score_fn.train()

    encoder_SI_prior.train()
    total_loss = 0
    criterion = nn.MSELoss()  # 均方误差损失

    for batch_idx, (data, relations) in enumerate(train_loader):
        data, relations = data.to(device), relations.to(device)
        optimizer_SIprior.zero_grad()
        h=emb_SI_prior(data)
        # **编码节点特征**
        scores = encoder_SI_prior(h,rel_rec, rel_send) # 形状: (batchsize, node_num, hidden_dim)
        # **归一化得到连边概率**
        edge_probs = F.softmax(scores,dim=-1)
       # 形状: (batchsize, node_num, node_num, num_edge_type=3)
        batchsize = data.size(0)

        # **重塑 relations 以匹配预测形状*
        mask_neg1 = (relations == 0) # 值为 -1 的位置
        mask_other = (relations == 1) | (relations == 1)  # 值为 0 或 1 的位置

        # 初始化输出张量
        batchsize, num_edges = relations.shape
        extended_relations = torch.zeros(batchsize, num_edges, 2, dtype=torch.float32, device=device)

        # 根据条件填充输出张量
        extended_relations[mask_neg1, :] = torch.tensor([1, 0], dtype=torch.float32, device=device)
        extended_relations[mask_other, :] = torch.tensor([0, 1], dtype=torch.float32, device=device)
        extended_relations = extended_relations.cuda()
        # extended_relations = torch.zeros((batchsize, 600, 3), dtype=torch.float32)
        # 根据 input_tensor 的值进行索引填充
        # # 注意：需要将 -1 映射到索引 0，0 映射到索引 1，1 映射到索引 2
        # indices = relations + 1  # 将 -1, 0, 1 转换为 0, 1, 2
        # extended_relations=extended_relations.cuda()
        # extended_relations.scatter_(
        #     dim=2,  # 在最后一个维度上操作
        #     index=indices.unsqueeze(-1),  # 将 indices 扩展为 [batchsize, num_edges, 1]
        #     src=torch.ones_like(extended_relations).cuda()  # 填充值为 1
        # )

        # **计算 MSE 损失**
        # print(edge_probs.shape)
        # print(extended_relations)
        # print(relations.shape)
        loss = criterion(edge_probs, extended_relations.float())  # 排除对角线的自环
        loss.backward()
        optimizer_SIprior.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.6f}")
        # if batch_idx % 100 == 0:
        #     pred_indices = torch.argmax(edge_probs, dim=-1)
        #
        #     # 将最大概率索引转换为 one-hot 形式
        #     pred_one_hot = torch.zeros_like(edge_probs)
        #     pred_one_hot.scatter_(-1, pred_indices.unsqueeze(-1), 1)
        #
        #     # 比较估算值和真实值是否相等
        #     correct_predictions = (pred_one_hot == extended_relations).all(dim=-1)
        #     accuracy = correct_predictions.float().mean().item()
        #
        #     print(f"Batch {batch_idx}/{len(train_loader)} - Acc: {accuracy:.6f}")

    return total_loss / len(train_loader)

def evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior,  data_loader, device,rel_rec, rel_send):
    emb_SI_prior.eval()
    encoder_SI_prior.eval()
    total_loss = 0
    total_correct = 0  # 记录所有 batch 内的正确预测数
    total_edges = 0  # 记录所有 batch 内的总边数

    criterion = nn.MSELoss()

    with torch.no_grad():
        for data, relations in data_loader:
            data, relations = data.to(device), relations.to(device)


            if data.shape[-2]>50:
                data=data[:,:,:49,:]

            h = emb_SI_prior(data)
            # **编码节点特征**
            # scores = encoder_SI_prior(h)
            scores = encoder_SI_prior(h, rel_rec, rel_send)  # 形状: (batchsize, node_num, hidden_dim)
            # **计算节点间相似性**

            # **归一化得到连边概率**
            edge_probs = F.softmax(scores,dim=-1)  # 形状: (batchsize, node_num, node_num, num_edge_type=3)
            batchsize = data.size(0)
            # **重塑 relations 以匹配预测形状**
            mask_neg1 = (relations == 0)  # 值为 -1 的位置
            mask_other = (relations == 1) | (relations == 1)  # 值为 0 或 1 的位置
            # 初始化输出张量
            batchsize, num_edges = relations.shape
            extended_relations = torch.zeros(batchsize, num_edges, 2, dtype=torch.float32, device=device)

            # 根据条件填充输出张量
            extended_relations[mask_neg1, :] = torch.tensor([1, 0], dtype=torch.float32, device=device)
            extended_relations[mask_other, :] = torch.tensor([0, 1], dtype=torch.float32, device=device)
            extended_relations = extended_relations.cuda()
            # extended_relations = torch.zeros((batchsize, 600, 3), dtype=torch.float32)
            # # 根据 input_tensor 的值进行索引填充
            # # 注意：需要将 -1 映射到索引 0，0 映射到索引 1，1 映射到索引 2
            # indices = relations + 1  # 将 -1, 0, 1 转换为 0, 1, 2
            # extended_relations = extended_relations.cuda()
            # extended_relations.scatter_(
            #     dim=2,  # 在最后一个维度上操作
            #     index=indices.unsqueeze(-1),  # 将 indices 扩展为 [batchsize, num_edges, 1]
            #     src=torch.ones_like(extended_relations).cuda()  # 填充值为 1
            # )
            indices = torch.clamp(relations , min=0, max=1)
            # **计算 MSE 损失**
            loss = criterion(edge_probs, extended_relations.float())  # 排除对角线的自环
            total_loss += loss.item()


            # 比较估算值和真实值是否相等
            pred_indices = torch.argmax(edge_probs, dim=-1)
            correct_predictions = (pred_indices == indices).sum().item()  # 计算所有正确预测的边数
            total_correct += correct_predictions
            total_edges += indices.numel()  # 计算所有边数

    accuracy = total_correct / total_edges
    return total_loss / len(data_loader), accuracy


# 主函数
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--save-probs', action='store_true', default=False,
    #                     help='Save the probs during test.')
    # parser.add_argument('--b-portion', type=float, default=1.0,
    #                     help='Portion of data to be used in benchmarking.')
    # parser.add_argument('--b-time-steps', type=int, default=49,
    #                     help='Portion of time series in data to be used in benchmarking.')
    # parser.add_argument('--b-shuffle', action='store_true', default=False,
    #                     help='Shuffle the data for benchmarking?.')
    # parser.add_argument('--b-manual-nodes', type=int, default=0,
    #                     help='The number of nodes if changed from the original dataset.')
    # parser.add_argument('--data-path', type=str, default='',
    #                     help='Where to load the data. May input the paths to edges_train of the data.')
    # parser.add_argument('--b-network-type', type=str, default='brain_networks',
    #                     help='What is the network type of the graph.')
    # parser.add_argument('--b-directed', action='store_true', default=True,
    #                     help='Default choose trajectories from undirected graphs.')
    # parser.add_argument('--b-simulation-type', type=str, default='netsims',
    #                     help='Either springs or netsims.')
    # parser.add_argument('--b-suffix', type=str, default='15r1',
    #                     help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
    #                          ' Or "50r1" for 50 nodes, rep 1 and noise free.')
    # parser.add_argument('--suffix', type=str, default='netsims',
    #                     help='Suffix for training data (e.g. "_charged".')
    # parser.add_argument('--batch-size', type=int, default=128,
    #                     help='Number of samples per batch.')
    parser.add_argument('--pseudo-label-save-folder', type=str, default='pseudo_data/',
                        help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--SI-prior-epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--SI-prior-hidden_dim', type=int, default=256,
                        help='SI-prior-hidden_dim.')
    parser.add_argument('--SI-prior-lr', type=int, default=0.0005,
                        help='SI-prior-learning-rate.')
    args = parser.parse_args()
    # **超参数**

    hidden_dim = args.SI_prior_hidden_dim
    num_epochs =args.SI_prior_epochs
    learning_rate = args.SI_prior_lr
    use_transformer = False  # 选择 RNN 还是 Transformer 作为编码器

    # **设备**
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_netsims_data_pse(args)
    train_loader2, valid_loader2, test_loader2, loc_max, loc_min, vel_max, vel_min = load_netsims_data(args)
    # train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_springs_datapse(args)
    # train_loader2, valid_loader2, test_loader2, loc_max, loc_min, vel_max, vel_min = load_springs_data(args)
    # emb_SI_prior = RNNemb_SI_prior(input_dim=dynamic_dim, hidden_dim=hidden_dim, rnn_type="GRU", bidirectional=False,dropout=0.0).to(device)
    emb_SI_prior = MLP_emb(n_in=49*1, n_hid=hidden_dim,n_out=hidden_dim,do_prob=0.0).to(device)
    encoder_SI_prior = MLPEncoderpre(49 * 1, hidden_dim,
                         2,
                         do_prob=0.0, factor=True).to(device)

    optimizer_SIprior=optim.Adam(
        list(emb_SI_prior.parameters()) + list(encoder_SI_prior.parameters()) ,
        lr=learning_rate
    , weight_decay=1e-5)

    # **初始化最佳验证准确率**
    best_val_acc = 0.0
    best_model_path = "SIprior_model/best_model.pth"

    # **验证初始模型**
    # val_loss, val_acc = evaluate_model(emb_SI_prior, score_fn, softmax_fn, valid_loader, device)
    # print(f"Initial Val Loss={val_loss:.6f}, Val Acc={val_acc:.4f}")

    # **训练循环**
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # **训练**
        train_loss = train_model(emb_SI_prior, encoder_SI_prior,  train_loader, optimizer_SIprior, device,rel_rec, rel_send)

        # **验证**
        val_loss, val_acc = evaluate_model(emb_SI_prior, encoder_SI_prior, valid_loader, device,rel_rec, rel_send)
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.4f}")
        val_loss2, val_acc2 = evaluate_model(emb_SI_prior, encoder_SI_prior,  valid_loader2, device, rel_rec, rel_send)
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss2={val_loss2:.6f}, Val Acc2={val_acc2:.4f}")

        # **检查是否保存模型**
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'emb_SI_prior_state_dict': emb_SI_prior.state_dict(),
                    'encoder_SI_prior_state_dict': encoder_SI_prior.state_dict(),

                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, best_model_path
            )
            print(f"==> New best model saved at epoch {epoch} with Val Acc2 {val_acc:.4f}")

    # **测试模型**
    test_loss, test_acc = evaluate_model(emb_SI_prior, encoder_SI_prior,  test_loader, device,rel_rec, rel_send)

    print(f"Final Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

    # **加载最佳模型进行最终测试**
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    emb_SI_prior.load_state_dict(checkpoint['emb_SI_prior_state_dict'])
    encoder_SI_prior.load_state_dict(checkpoint['encoder_SI_prior_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    test_loss, test_acc = evaluate_model(emb_SI_prior, encoder_SI_prior,  test_loader, device,rel_rec, rel_send)

    print(f"Best Model Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

if __name__ == "__main__":
    main()
