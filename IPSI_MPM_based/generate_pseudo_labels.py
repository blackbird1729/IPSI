import torch
import config as cfg
from argparse import ArgumentParser
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from generate.load import load_kuramoto, load_nri, load_netsims, load_nri_benchmark, load_netsims_benchmark
import os
from utils.load_data import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset

class DataWrapper(Dataset):
    """
    A wrapper for torch.utils.data.Dataset.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

def make_pseudo_labels(args):
    def load_data(inputs, batch_size: int, shuffle: bool=True):
        """
        Return a dataloader given the input and the batch size.
        """
        data = DataWrapper(inputs)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle)
        return batches
    # 1. 加载数据
    # train_loader, valid_loader, test_loader = \
    #     load_kuramoto_data(batch_size=args.batch_size, suffix=args.suffix)
    data = load_customized_data(args)

    if args.dyn == 'kuramoto':
        data, es, _ = load_kuramoto(data, args.size)

    # original:
    elif args.dyn == 'springs':
        data, es, _ = load_nri(data, args.size)


    # modification for benchmark:
    # elif args.dyn == 'springs':
    #     data, es, _ = load_nri_benchmark(data, args.size, args)

    # original:
    else:
        data, es, _ = load_netsims(data, args.size)
    # modification for benchmark:
    # else:
    #     data, es, _ = load_netsims_benchmark(data, args.size, args)
    if args.data_path != '':
        dim = args.dim if args.reduce == 'cnn' else args.dim * args.b_time_steps
    else:
        dim = args.dim if args.reduce == 'cnn' else args.dim * cfg.train_steps
    encs = {
        'GNNENC': GNNENC,
        'RNNENC': RNNENC,
        'AttENC': AttENC,
    }
    decs = {
        'GNNDEC': GNNDEC,
        'RNNDEC': RNNDEC,
        'AttDEC': AttDEC,
    }
    encoder = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, cfg.do_prob_enc, reducer=args.reduce)
    decoder = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, cfg.do_prob_dec,
                             skip_first=args.skip)
    model = NRIModel(encoder, decoder, es, args.size)

    name = 'baseline_data/best.pth'
    model.load_state_dict(torch.load(name))
    # model = DataParallel(model)
    if cfg.gpu:
        model = model.cuda()

    data = {key: TensorDataset(value[0], value[1])
            for key, value in data.items()}

    def extract_edges_single(model, data):
        """
        为每个数据集（train, val, test）生成一个单一的、基于所有样本平均概率的结构伪标签，
        并将其转换为填充了0对角线的邻接矩阵。

        Args:
            model: 训练好的模型，具有 predict_relations 方法。
            data: 包含 'train', 'val', 'test' 键的字典，值用于加载数据。

        Returns:
            pseudo_adj_mats: 字典，键为 'train', 'val', 'test'，值为对应的邻接矩阵 [N, N]。
        """
        pseudo_adj_mats = {}
        # 假设 N (节点数) 可以从数据或模型中获取。这里假设我们知道 N 或可以从第一个 batch 推断。
        # N = ... # 需要根据实际情况确定 N

        for split in ['train', 'val', 'test']:
            print(f"Processing {split} split...")
            prob_list = []  # 用于累积所有批次的 prob (在 B 维度上)
            data_now = load_data(data[split], 128)  # batch_size=128

            # --- 第一步：累积整个数据集的 prob ---
            for batch_idx, (_, states) in enumerate(data_now):
                states = states.to('cuda')  # [B, T, N, D]
                states = states[:, :cfg.train_steps, :, :]  # [B, T_trimmed, N, D]
                # print(f"Batch {batch_idx}, states shape: {states.shape}")

                with torch.no_grad():
                    # 假设 model.predict_relations(states) 输出 [E, B, D]
                    prob = model.predict_relations(states)  # [E, B, D]
                    # print(f"prob shape: {prob.shape}")
                    prob_list.append(prob.cpu())  # 移到CPU，节省GPU内存

            # 将所有批次的 prob 在批次维度 (dim=1) 上拼接
            # prob_all 形状: [E, Total_B, D]
            prob_all = torch.cat(prob_list, dim=1)
            print(f"Combined prob shape for {split}: {prob_all.shape}")

            # --- 第二步：计算所有样本上的平均概率 ---
            # 在样本/批次维度 (dim=1) 上求平均，得到 [E, D]
            # 这相当于对每个边和每个状态类别，计算所有样本的平均概率。
            prob_mean = prob_all.mean(dim=1)  # [E, D]
            print(f"Mean prob shape for {split}: {prob_mean.shape}")

            # --- 第三步：取 argmax 得到单一的边状态标签 ---
            # 对平均概率在状态类别维度 (dim=-1 or dim=1) 取 argmax
            # 得到最终的单一伪边标签，形状 [E]
            single_edges = prob_mean.argmax(dim=-1).numpy()  # [E]
            print(f"Single pseudo edges shape for {split}: {single_edges.shape}")

            # --- 第四步：转换为邻接矩阵 ---
            # 需要知道 N。假设 E = N*(N-1)。可以从 single_edges 的长度推断 N。
            E = single_edges.shape[0]
            # 解方程 E = N*(N-1) => N^2 - N - E = 0
            # N = (1 + sqrt(1 + 4*E)) / 2
            N_float = (1 + np.sqrt(1 + 4 * E)) / 2
            N = int(round(N_float))
            assert N * (N - 1) == E, f"Inferred N={N} does not satisfy E=N*(N-1)={N * (N - 1)} for E={E}"
            print(f"Inferred number of nodes N for {split}: {N}")

            # 创建 [N, N] 的邻接矩阵
            adj_mat = np.zeros((N, N), dtype=single_edges.dtype)  # 用0初始化，对角线自然为0

            # 填充非对角线元素
            # 假设边的顺序是按行优先排列的 (i, j), i != j
            # 即： (0,1), (0,2), ..., (0, N-1), (1,0), (1,2), ..., (1, N-1), ..., (N-1, 0), ..., (N-1, N-2)
            edge_idx = 0
            for i in range(N):
                for j in range(N):
                    if i != j:  # 跳过对角线
                        adj_mat[i, j] = single_edges[edge_idx]
                        edge_idx += 1

            assert edge_idx == E, f"Filled {edge_idx} edges, expected {E}"

            pseudo_adj_mats[split] = adj_mat
            print(f"Generated adjacency matrix for {split}: {adj_mat.shape}")

        return pseudo_adj_mats

    def save_edges(pseudo_edges, suffix='_pseudo'):

        # os.makedirs('data/kuramoto/', exist_ok=True)
        for split in ['train', 'val', 'test']:
            path = 'pseudo_data/edges_'.format(args.dyn) + split + '_pseudo.npy'
            np.save(path, pseudo_edges[split])
        print("Pseudo-labels saved to data/spring/")

    pseudo_labels = extract_edges_single(model, data)
    save_edges(pseudo_labels, suffix='_pseudo')

if __name__ == '__main__':
    def init_args():
        parser = ArgumentParser()
        parser.add_argument('--dyn', type=str, default='',
                            help='Type of dynamics: springs, charged, kuramoto or netsims.')
        parser.add_argument('--size', type=int, default=5,
                            help='Number of particles.')
        parser.add_argument('--dim', type=int, default=4,
                            help='Dimension of the input states.')
        parser.add_argument('--epochs', type=int, default=500,
                            help='Number of training epochs. 0 for testing.')
        parser.add_argument('--reg', type=float, default=0,
                            help='Penalty factor for the symmetric prior.')
        parser.add_argument('--batch', type=int, default=2 ** 6, help='Batch size.')
        parser.add_argument('--skip', action='store_true', default=True,
                            help='Skip the last type of edge.')
        parser.add_argument('--no_reg', action='store_true', default=False,
                            help='Omit the regularization term when using the loss as an validation metric.')
        parser.add_argument('--sym', action='store_true', default=False,
                            help='Hard symmetric constraint.')
        parser.add_argument('--reduce', type=str, default='mlp',
                            help='Method for relation embedding, mlp or cnn.')
        parser.add_argument('--enc', type=str, default='RNNENC', help='Encoder.')
        parser.add_argument('--dec', type=str, default='RNNDEC', help='Decoder.')
        parser.add_argument('--scheme', type=str, default='both',
                            help='Training schemes: both, enc or dec.')
        parser.add_argument('--load_path', type=str, default='',
                            help='Where to load a pre-trained model.')
        parser.add_argument('--save_folder', type=str, default='logs',
                            help='Where to save the trained model, leave empty to not save anything.')

        # for benchmark:
        parser.add_argument('--data_path', type=str, default='',
                            help='Where to load the data. May input the paths to edges_train of the data.')
        parser.add_argument('--save-probs', action='store_true', default=False,
                            help='Save the probs during test.')
        parser.add_argument('--b-network-type', type=str, default='vascular_networks',
                            help='What is the network type of the graph.gene_regulatory_networks or vascular_networks')
        parser.add_argument('--b-directed', action='store_true', default=True,
                            help='Default choose trajectories from undirected graphs.')
        parser.add_argument('--b-simulation-type', type=str, default='springs',
                            help='Either springs or netsims.')
        parser.add_argument('--b-suffix', type=str, default='15r1',
                            help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
                                 ' Or "50r1" for 50 nodes, rep 1 and noise free.')
        parser.add_argument('--b-portion', type=float, default=1.0,
                            help='Portion of data to be used in benchmarking.')
        parser.add_argument('--b-time-steps', type=int, default=49,
                            help='Portion of time series in data to be used in benchmarking.')
        parser.add_argument('--b-shuffle', action='store_true', default=False,
                            help='Shuffle the data for benchmarking?.')
        parser.add_argument('--b-manual-nodes', type=int, default=0,
                            help='The number of nodes if changed from the original dataset.')
        parser.add_argument('--seed', type=int, default=42, help='Random seed.')
        # remember to disable this for submission
        parser.add_argument('--b-walltime', action='store_true', default=True,
                            help='Set wll time for benchmark training and testing. (Max time = 2 days)')
        args = parser.parse_args()
        if args.dyn == '':
            args.dyn = args.b_simulation_type

        if args.data_path == "" and args.b_network_type != "":
            if args.b_directed:
                dir_str = 'directed'
            else:
                dir_str = 'undirected'
            args.data_path = os.path.dirname(
                os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                             dir_str + \
                             '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
            args.b_manual_nodes = int(args.b_suffix.split('r')[0])
        if args.data_path != '':
            args.size = args.b_manual_nodes

        if args.b_simulation_type == 'springs':
            args.dim = 4
        elif args.b_simulation_type == 'netsims':
            args.dim = 1

        if args.b_time_steps < 49:
            args.reduce = 'mlp'

        return args
    args = init_args()
    cfg.init_args(args)
    if args.dyn == 'kuramoto':
        args.skip = True
        args.dim = 3
    else:
        args.skip = False
        args.dim = 4

    make_pseudo_labels(args)