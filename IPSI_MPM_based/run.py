from operator import truediv

import torch
import config as cfg
from src.models.MPM.generate_pseudo_labels import make_pseudo_labels
from instructors.XNRI import XNRIIns
from instructors.XNRI_enc import XNRIENCIns
from instructors.XNRI_dec import XNRIDECIns
from argparse import ArgumentParser
from utils.load_data import *
from models.encoder import AttENC, RNNENC, GNNENC
from models.decoder import GNNDEC, RNNDEC, AttDEC
from models.nri import NRIModel
from torch.nn.parallel import DataParallel
from generate.load import load_kuramoto, load_nri, load_netsims
from check_acc import compute_accuracy
from models.encoder_joint import RNNENC_joint
from models.nri_joint import NRIModel_joint
from instructors.XNRI_joint import XNRIIns_joint
import numpy as np
import os
import datetime
import time


def init_args():
    parser = ArgumentParser()
    parser.add_argument('--dyn', type=str, default='',
    help='Type of dynamics: springs, charged, kuramoto or netsims.')
    parser.add_argument('--size', type=int, default=5, 
    help='Number of particles.')
    parser.add_argument('--dim', type=int, default=4, 
    help='Dimension of the input states.')
    parser.add_argument('--epochs', type=int, default=10,
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
    parser.add_argument('--save_folder', type=str, default='baseline_data',
                        help='Where to save the trained model, leave empty to not save anything.')

    # for benchmark:
    parser.add_argument('--data_path', type=str, default='',
    help='Where to load the data. May input the paths to edges_train of the data.')
    parser.add_argument('--save-probs', action='store_true', default=False,
                        help='Save the probs during test.')
    parser.add_argument('--b-network-type', type=str, default='vascular_networks',
                        help='What is the network type of the graph.')
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
    parser.add_argument('--pseudo-path', type=str, default="pseudo_data/edges_test_pseudo.npy",
                        help='Path to the pseudo label .npy file')
    args = parser.parse_args()
    args.exclude_diagonal = False
    if args.dyn == '':
        args.dyn = args.b_simulation_type

    if args.data_path == "" and args.b_network_type != "":
        if args.b_directed:
            dir_str = 'directed'
        else:
            dir_str = 'undirected'
        args.data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
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





def run():
    args = init_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cfg.gpu:
        torch.cuda.manual_seed(args.seed)

    # load data
    # original:
    # data = load_data(args)
    start_time = time.time()
    # customized pipeline for saving:
    name_str = args.data_path.split('/')[-3] + '_' + args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    save_folder = './'+args.save_folder+'/'
    # save_folder = './{}/MPM-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.enc,
    #                                                   args.dec, timestamp)
    save_folder = save_folder.replace(":", "_")
    # os.mkdir(save_folder)
    args.save_folder = save_folder
    cfg.init_args(args)
    res_folder = save_folder + 'results/'
    os.makedirs('./baseline_data/results/', exist_ok=True)

    # customized:
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
    if args.load_path:
        name = 'logs/{}/best.pth'.format(args.load_path)
        model.load_state_dict(torch.load(name))
    model = DataParallel(model)
    if cfg.gpu:
        model = model.cuda()
    if args.scheme == 'both':
        # Normal training.
        ins = XNRIIns(model, data, es, args)
    elif args.scheme == 'enc':
        # Only train the encoder.
        ins = XNRIENCIns(model, data, es, args)
    elif args.scheme == 'dec':
        # Only train the decoder.
        ins = XNRIDECIns(model, data, es, args)
    else:
        raise NotImplementedError('training scheme: both, enc or dec')
    ins.train(save_folder, start_time)
    print("Finished.")
    print("Dataset: ", args.dyn)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")

    make_pseudo_labels(args)
    pseudo = np.load(args.pseudo_path)  # [num_nodes, num_nodes]
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    true = np.load(root_str + 'edges_test_' + keep_str)
    print("Pseudo label:\n", pseudo)
    print("True label:\n", true)
    print(f"Loaded pseudo: {pseudo.shape}, true: {true.shape}")
    assert pseudo.shape == true.shape, "Shape mismatch between prediction and ground truth!"
    assert pseudo.shape[0] == pseudo.shape[1], "Matrix must be square for diagonal exclusion."
    acc = compute_accuracy(pseudo, true, exclude_diagonal=args.exclude_diagonal)
    diag_msg = " (excluding diagonal)" if args.exclude_diagonal else " (including diagonal)"
    print(f"Accuracy{diag_msg}: {acc * 100:.2f}%")

    save_folder = './SIprior_data/'
    args.save_folder = save_folder
    cfg.init_args(args)
    res_folder = save_folder + 'results/'
    os.makedirs('./SIprior_data/results/', exist_ok=True)

    data_SIprior = load_customized_data_pse(args)
    if args.dyn == 'kuramoto':
        data_SIprior, es, _ = load_kuramoto(data_SIprior, args.size)

    # original:
    elif args.dyn == 'springs':
        data_SIprior, es, _ = load_nri(data_SIprior, args.size)

    else:
        data_SIprior, es, _ = load_netsims(data_SIprior, args.size)
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
    encoder_SIprior = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, cfg.do_prob_enc, reducer=args.reduce)
    decoder_SIprior = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, cfg.do_prob_dec,
                             skip_first=args.skip)
    model_SIprior = NRIModel(encoder_SIprior, decoder_SIprior, es, args.size)
    model_SIprior = DataParallel(model_SIprior)
    if cfg.gpu:
        model_SIprior = model_SIprior.cuda()
    ins_SIprior = XNRIENCIns(model_SIprior, data_SIprior, es, args)
    epochs=args.epochs
    args.epochs = 100
    ins_SIprior.train(save_folder)
    args.epochs = epochs
    save_folder = './SIjoint_data/'
    args.save_folder = save_folder
    cfg.init_args(args)
    res_folder = save_folder + 'results/'
    os.makedirs('./SIprior_data/results/', exist_ok=True)

    encoder_prior = encs[args.enc](dim, cfg.n_hid, cfg.edge_type, cfg.do_prob_enc, reducer=args.reduce)
    encoder_joint = RNNENC_joint(dim, cfg.n_hid, cfg.edge_type, cfg.do_prob_enc, reducer=args.reduce)
    decoder = decs[args.dec](args.dim, cfg.edge_type, cfg.n_hid, cfg.n_hid, cfg.n_hid, cfg.do_prob_dec,
                             skip_first=args.skip)
    model_prior = NRIModel(encoder_prior, decoder, es, args.size)
    model_joint = NRIModel_joint(encoder_joint, decoder, es, args.size)

    name = ('SIprior_data/best.pth')
    model_prior.load_state_dict(torch.load(name))

    # model_prior = DataParallel(model_prior)
    model_joint = DataParallel(model_joint)
    if cfg.gpu:
        model_prior = model_prior.cuda()
        model_joint = model_joint.cuda()

    ins_joint = XNRIIns_joint(model_joint, model_prior, data, es, args)
    ins_joint.train(save_folder, start_time)
    print("Finished.")
    print("Dataset: ", args.dyn)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")

#
if __name__ == "__main__":
    run()
