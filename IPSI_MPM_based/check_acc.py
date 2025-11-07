import numpy as np
import torch
import argparse
import os

def compute_accuracy(pseudo, true, exclude_diagonal=False):
    """
    Compute element-wise accuracy between two [num_nodes, num_nodes] matrices.

    Args:
        pseudo: np.ndarray, predicted labels, shape [num_nodes, num_nodes]
        true: np.ndarray, ground truth labels, shape [num_nodes, num_nodes]
        exclude_diagonal (bool): whether to exclude diagonal elements (self-loops) from calculation

    Returns:
        acc: torch.Tensor, scalar accuracy (0.0 ~ 1.0)
    """
    # Convert to torch tensors
    pred_tensor = torch.from_numpy(pseudo).to(dtype=torch.int64)
    true_tensor = torch.from_numpy(true).to(dtype=torch.int64)

    # Create mask for comparison
    if exclude_diagonal:
        # Create a mask where diagonal is False, others are True
        mask = ~torch.eye(pred_tensor.size(0), dtype=torch.bool, device=pred_tensor.device)
        correct = ((pred_tensor == true_tensor) & mask).sum()
        total = mask.sum()
    else:
        correct = (pred_tensor == true_tensor).sum()
        total = true_tensor.numel()

    acc = correct.float() / total
    return acc


def main(args):
    # Load pseudo and true labels
    pseudo = np.load(args.pseudo_path)  # [num_nodes, num_nodes]
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    true = np.load(root_str + 'edges_test_' + keep_str)
    # true = np.load(args.true_path)  # [num_nodes, num_nodes]

    print("Pseudo label:\n", pseudo)
    print("True label:\n", true)
    print(f"Loaded pseudo: {pseudo.shape}, true: {true.shape}")

    # Validate shape
    assert pseudo.shape == true.shape, "Shape mismatch between prediction and ground truth!"

    # Check if shapes are square (required for diagonal exclusion)
    assert pseudo.shape[0] == pseudo.shape[1], "Matrix must be square for diagonal exclusion."

    # Compute accuracy
    acc = compute_accuracy(pseudo, true, exclude_diagonal=args.exclude_diagonal)
    diag_msg = " (excluding diagonal)" if args.exclude_diagonal else " (including diagonal)"
    print(f"Accuracy{diag_msg}: {acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check pseudo-label accuracy")
    parser.add_argument('--pseudo-path', type=str, default="pseudo_data/edges_test_pseudo.npy",
                        help='Path to the pseudo label .npy file')
    parser.add_argument('--data_path', type=str, default='',
                        help='Where to load the data. May input the paths to edges_train of the data.')
    parser.add_argument('--b-network-type', type=str, default='vascular_networks',
                        help='What is the network type of the graph.')
    parser.add_argument('--b-directed', action='store_true', default=True,
                        help='Default choose trajectories from undirected graphs.')
    parser.add_argument('--b-simulation-type', type=str, default='springs',
                        help='Either springs or netsims.')
    parser.add_argument('--b-suffix', type=str, default='15r1',
                        help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
                             ' Or "50r1" for 50 nodes, rep 1 and noise free.')
    args = parser.parse_args()
    args.exclude_diagonal = False
    if args.data_path == "" and args.b_network_type != "":
        if args.b_directed:
            dir_str = 'directed'
        else:
            dir_str = 'undirected'
        args.data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                         dir_str + \
                         '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
        args.b_manual_nodes = int(args.b_suffix.split('r')[0])
    main(args)