import torch
import os
import numpy as np
import argparse


def compute_accuracy(pseudo, true):
    """
    pseudo, true: [num_nodes, num_nodes] (shared for all samples)
    """
    # 转为 torch 张量
    pred_tensor = torch.from_numpy(pseudo)
    true_tensor = torch.from_numpy(true)

    # 确保数据类型一致并比较
    pred_tensor = pred_tensor.to(dtype=torch.int64)
    true_tensor = true_tensor.to(dtype=torch.int64)

    # 逐元素比较
    correct = (pred_tensor == true_tensor).sum()
    total = true_tensor.numel()
    acc = correct.float() / total
    return acc


def check_pseudo_acc(args):
    # 加载伪标签和真实标签
    if args.b_directed:
        dir_str = 'directed'
    else:
        dir_str = 'undirected'
    pseudo = np.load(args.pseudo_label_save_folder+'/edges_test'+args.b_simulation_type+'pse.npy')  # [num_nodes, num_nodes]
    true = np.load(os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                     dir_str +\
                     '/' + args.b_simulation_type + '/edges_test_' + args.b_simulation_type + args.b_suffix + '.npy')  # [num_nodes, num_nodes]

    print("Pseudo label:\n", pseudo)
    print("True label:\n", true)
    print(f"Loaded pseudo: {pseudo.shape}, true: {true.shape}")

    # 验证形状一致性
    assert pseudo.shape == true.shape, "Shape mismatch between prediction and ground truth!"

    # 计算准确率
    acc = compute_accuracy(pseudo, true)
    print(f"Pseudo Label Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pseudo-label accuracy")
    parser.add_argument('--pseudo-label-save-folder', type=str, default="pseudo_data/",
                        help='Path to the pseudo label .npy file')
    parser.add_argument('--true-path', type=str, default="pseudo_data/edges_test_netsims15r1.npy",
                        help='Path to the ground truth .npy file')
    args = parser.parse_args()
    check_pseudo_acc(args)
