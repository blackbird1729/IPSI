# pseudo_label.py

import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable

from utils import *
from modules_baseline import MLPEncoder, CNNEncoder  # 根据你训练时选择的 encoder 类型导入


def make_pseudo_labels(rel_rec,rel_send,train_loader,valid_loader,test_loader,args):
    # 1. 加载数据
    # train_loader, valid_loader, test_loader = \
    #     load_kuramoto_data(batch_size=args.batch_size, suffix=args.suffix)
    # if args.suffix == "springs":
    #     train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_springs_data(
    #         args)
    # else:
    #     train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_netsims_data(
    #         args)
    # # 2. 构建 off_diag, rel_rec, rel_send
    # num_atoms = args.num_atoms
    # off_diag = np.ones([num_atoms, num_atoms])
    # rel_rec = torch.FloatTensor(np.array(
    #     encode_onehot(np.where(off_diag)[0]), dtype=np.float32))
    # rel_send = torch.FloatTensor(np.array(
    #     encode_onehot(np.where(off_diag)[1]), dtype=np.float32))
    # if args.cuda:
    #     rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
    # rel_rec = Variable(rel_rec)
    # rel_send = Variable(rel_send)

    # 3. 初始化 encoder 并加载权重
    if args.encoder == 'mlp':
        encoder = MLPEncoder(args.timesteps * args.dims,
                             args.encoder_hidden,
                             args.edge_types,
                             args.encoder_dropout,
                             factor=not args.no_factor)
    else:
        encoder = CNNEncoder(args.dims,
                             args.encoder_hidden,
                             args.edge_types,
                             args.encoder_dropout,
                             factor=not args.no_factor)

    ckpt = torch.load(os.path.join(args.baseline_model_load_folder, 'encoder.pt'),
                      map_location='cuda' if args.cuda else 'cpu')
    encoder.load_state_dict(ckpt)
    encoder.eval()
    if args.cuda:
        encoder.cuda()

    # 4. 遍历每个 split，生成伪标签
    # rows, cols = np.where(off_diag)
    rows, cols = np.indices((args.num_atoms, args.num_atoms))
    rows = rows.flatten()
    cols = cols.flatten()

    def infer_split(loader, split_name):
        num_nodes = args.num_atoms
        num_edges = num_nodes * (num_nodes )

        total_logits = torch.zeros(num_edges, args.edge_types, device='cuda' if args.cuda else 'cpu')
        total_samples = 0

        with torch.no_grad():
            for feat, relations in loader:
                if args.cuda:
                    feat = feat.cuda()
                    relations = relations.cuda()
                feat = Variable(feat)

                if split_name == "test":
                    feat_encoder = feat[:, :, :args.timesteps, :].contiguous()
                    logits = encoder(feat_encoder, rel_rec, rel_send)
                else:
                    logits = encoder(feat, rel_rec, rel_send)
                # print(logits.shape)
                # print(relations.shape)
                total_logits += logits.sum(dim=0)  # sum over batch dimension
                total_samples += logits.size(0)
                acc = edge_accuracy(logits, relations)

                # print(acc)
        avg_logits = total_logits / total_samples
        # print(avg_logits)
        preds = avg_logits.argmax(dim=-1)  # shape: [num_edges]
        # print(preds)
        # 构造邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=preds.device)
        adj[rows, cols] = preds.float()

        # adj[torch.arange(num_nodes), torch.arange(num_nodes)] = 0  # clear diagonal

        # adj[(adj == 0) & (off_diag == 1)] = -1  # 可选：替换非连接为 -1

        adj_np = adj.cpu().numpy()
        out_path = os.path.join(args.pseudo_label_save_folder, f'edges_{split_name}{args.suffix}pse.npy')
        np.save(out_path, adj_np)
        print(f'[{split_name}] saved {adj_np.shape} → {out_path}')



    os.makedirs(args.pseudo_label_save_folder, exist_ok=True)
    infer_split(train_loader, 'train')
    infer_split(valid_loader, 'valid')
    infer_split(test_loader,  'test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate pseudo-edge labels via pretrained encoder')
    parser.add_argument('--baseline-model-load-folder', type=str, default= 'baseline_model',
                        help='训练好的模型目录，包含 encoder.pt')
    parser.add_argument('--pseudo-label-save-folder', type=str, default='pseudo_data',
                        help='输出伪标签 .npy 文件的保存目录')
    parser.add_argument('--suffix', type=str, default='netsims',
                        help='数据集后缀，例如 "_charged5"')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timesteps', type=int, default=49)
    parser.add_argument('--dims', type=int, default=1)
    parser.add_argument('--num-atoms', type=int, default=15)
    parser.add_argument('--encoder', type=str, default='mlp',
                        choices=['mlp', 'cnn'])
    parser.add_argument('--encoder-hidden', type=int, default=256)
    parser.add_argument('--encoder-dropout', type=float, default=0.0)
    parser.add_argument('--edge-types', type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.5,
                        help='Gumbel-Softmax temperature')
    parser.add_argument('--no-factor', action='store_true', default=False,
                        help='Disables factor graph model.')

    parser.add_argument('--save-probs', action='store_true', default=False,
                        help='Save the probs during test.')
    parser.add_argument('--b-portion', type=float, default=1.0,
                        help='Portion of data to be used in benchmarking.')
    parser.add_argument('--b-time-steps', type=int, default=49,
                        help='Portion of time series in data to be used in benchmarking.')
    parser.add_argument('--b-shuffle', action='store_true', default=False,
                        help='Shuffle the data for benchmarking?.')
    parser.add_argument('--b-manual-nodes', type=int, default=0,
                        help='The number of nodes if changed from the original dataset.')
    parser.add_argument('--data-path', type=str, default='pseudo_data/edges_test_netsims15r1.npy',
                        help='Where to load the data. May input the paths to edges_train of the data.')
    parser.add_argument('--b-network-type', type=str, default='brain_networks',
                        help='What is the network type of the graph.')
    parser.add_argument('--b-directed', action='store_true', default=True,
                        help='Default choose trajectories from undirected graphs.')
    parser.add_argument('--b-simulation-type', type=str, default='netsims',
                        help='Either springs or netsims.')
    parser.add_argument('--b-suffix', type=str, default='15r1',
                        help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
                             ' Or "50r1" for 50 nodes, rep 1 and noise free.')
    args = parser.parse_args()
    args.cuda = (not args.no_cuda) and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.data_path == "" and args.b_network_type != "":
        if args.b_directed:
            dir_str = 'directed'
        else:
            dir_str = 'undirected'
        args.data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                         dir_str + \
                         '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
        args.b_manual_nodes = int(args.b_suffix.split('r')[0])


    make_pseudo_labels(args)
    print(args.data_path)