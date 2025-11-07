from __future__ import division
from __future__ import print_function

import numpy as np
import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score

from src.models.NRI.train_prior import train_SIprior_model, evaluate_SIpriormodel
from src.models.NRI.utils import load_springs_data_pse
from utils import *
from modules_baseline import *
from baseline import train_baseline,test_baseline
from generate_pseudo_labels import make_pseudo_labels
from generate_pseudo_labels_ite import make_pseudo_labels_iteratively
from check_pse_acc import check_pseudo_acc
from modules_SI_prior import *
from train_joint import train_SIjoint,test_SIjoint
from modules_SI_joint import MLPEncoder_SIjoint,MLPDecoder_SIjoint
t_begin = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=15,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp, cnn, or gin).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

# for benchmark:
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
parser.add_argument('--data-path', type=str, default='',
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
# remember to disable this for submission
parser.add_argument('--b-walltime', action='store_true', default=True,
                    help='Set wll time for benchmark training and testing. (Max time = 2 days)')

#for IPSI-SI:
parser.add_argument('--baseline-model-load-folder', type=str, default= 'baseline_model',
                        help='训练好的模型目录，包含 encoder.pt')
parser.add_argument('--pseudo-label-save-folder', type=str, default='pseudo_data',
                        help='输出伪标签 .npy 文件的保存目录')
#for train SIprior

parser.add_argument('--SI-prior-epochs', type=int, default=100,
                    help='Number of epochs to train SI-prior.')
parser.add_argument('--SI-joint-epochs', type=int, default=250,
                    help='Number of epochs to train SI-joint.')
parser.add_argument('--SI-prior-hidden_dim', type=int, default=256,
                    help='SI-prior-hidden_dim.')
parser.add_argument('--SI-prior-lr', type=int, default=0.0005,
                    help='SI-prior-learning-rate.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

if args.suffix == "":
    args.suffix = args.b_simulation_type
    args.timesteps = args.b_time_steps

if args.b_simulation_type == 'springs':
    args.dims = 4
elif args.b_simulation_type == 'netsims':
    args.dims = 1

if args.data_path == "" and args.b_network_type != "":
    if args.b_directed:
        dir_str = 'directed'
    else:
        dir_str = 'undirected'
    args.data_path = os.path.dirname(os.path.dirname(os.getcwd())) + '/simulations/' + args.b_network_type + '/' + \
                     dir_str +\
                     '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
    args.b_manual_nodes = int(args.b_suffix.split('r')[0])
if args.data_path != '':
    args.num_atoms = args.b_manual_nodes
# if args.data_path != '':
#     args.suffix = args.data_path.split('/')[-1].split('_', 2)[-1]

print("suffix: ", args.suffix)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat().replace(":", "-").split(".")[0]
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
               args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
    # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    # save_folder = './{}/NRI-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.encoder,
    #                                                   args.decoder, timestamp)
    save_folder = 'baseline_model/'
    # os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    res_folder = save_folder + 'results/'
    # os.mkdir(res_folder)
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.prediction_steps > args.timesteps:
    args.prediction_steps = args.timesteps

if args.suffix == "springs":
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_springs_data(
        args)
else:
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_netsims_data(
        args)

# original:
# train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
#     args.batch_size, args.suffix)

# Generate off-diagonal interaction graph: discarded
# off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
print("num_atoms: ", args.num_atoms)
off_diag = np.ones([args.num_atoms, args.num_atoms])

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'gin':
    encoder = GINEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

# if args.load_folder:
#     encoder_file = os.path.join(args.load_folder, 'encoder.pt')
#     encoder.load_state_dict(torch.load(encoder_file))
#     decoder_file = os.path.join(args.load_folder, 'decoder.pt')
#     decoder.load_state_dict(torch.load(decoder_file))
#
#     args.save_folder = False

optimizer_baseline = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler_baseline = lr_scheduler.StepLR(optimizer_baseline, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

#######################################
# Train baseline model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    t_epoch_start = time.time()
    val_loss = train_baseline(epoch, encoder,decoder,scheduler_baseline,train_loader,optimizer_baseline,rel_rec,rel_send,valid_loader,encoder_file,decoder_file,log,best_val_loss,args)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - t_epoch_start
    if args.b_walltime:
        if epoch_end_time - t_begin < 171900 - epoch_time:
            continue
        else:
            break
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test_baseline(encoder,decoder,rel_rec,rel_send,test_loader,encoder_file,decoder_file,save_folder,log,args)
if log is not None:
    print(save_folder)
    log.close()

# ###################
print("Generating pseudo labels using baseline models")
make_pseudo_labels(rel_rec,rel_send,train_loader,valid_loader,test_loader,args)
print(args.data_path)
# ###################
print("Testing pseudo labels accuracy")
check_pseudo_acc(args)
#
# ###################
print("Training SIprior in the first round")
if args.suffix == "springs":
    train_loader_SI_prior, valid_loader_SI_prior, test_loader_SI_prior, loc_max, loc_min, vel_max, vel_min = load_springs_data_pse(
        args)
else:
    train_loader_SI_prior, valid_loader_SI_prior, test_loader_SI_prior, loc_max, loc_min, vel_max, vel_min = load_netsims_data_pse(args)

emb_SI_prior = MLP_emb(n_in=args.timesteps*args.dims, n_hid=args.SI_prior_hidden_dim,n_out=args.SI_prior_hidden_dim,do_prob=0.0).to(device)
encoder_SI_prior = MLPEncoderpre(args.timesteps * args.dims, args.SI_prior_hidden_dim,
                         args.edge_types,
                         do_prob=0.0, factor=True).to(device)
optimizer_SIprior=optim.Adam(
        list(emb_SI_prior.parameters()) + list(encoder_SI_prior.parameters()) ,
        lr=args.SI_prior_lr
    , weight_decay=1e-5)
best_val_acc = 0.0
best_model_path = "SIprior_model/SIprior_best_model.pth"
for epoch in range(1, args.SI_prior_epochs+ 1):
    print(f"Epoch {epoch}/{args.SI_prior_epochs}")
    train_loss = train_SIprior_model(emb_SI_prior, encoder_SI_prior, train_loader_SI_prior, optimizer_SIprior, device, rel_rec, rel_send)
    val_loss, val_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, valid_loader_SI_prior, device, rel_rec, rel_send)
    print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.4f}")
    val_loss2, val_acc2 = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, valid_loader, device, rel_rec, rel_send)
    print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss2={val_loss2:.6f}, Val Acc2={val_acc2:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {
                'epoch': epoch,
                'emb_SI_prior_state_dict': emb_SI_prior.state_dict(),
                'encoder_SI_prior_state_dict': encoder_SI_prior.state_dict(),
                'optimizer_state_dict': optimizer_SIprior.state_dict(),
                'val_acc': val_acc,
            }, best_model_path
        )
        print(f"==> New best model saved at epoch {epoch} with Val Acc2 {val_acc:.4f}")
test_loss, test_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, test_loader_SI_prior, device, rel_rec, rel_send)

print(f"Final Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

print("Loading best model for evaluation on original data")
checkpoint = torch.load(best_model_path)
emb_SI_prior.load_state_dict(checkpoint['emb_SI_prior_state_dict'])
encoder_SI_prior.load_state_dict(checkpoint['encoder_SI_prior_state_dict'])
optimizer_SIprior.load_state_dict(checkpoint['optimizer_state_dict'])
test_loss, test_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, test_loader, device, rel_rec, rel_send)
print(f"Best Model Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

#############################
print("Training SIjoint in the first round")

encoder_SI_joint = MLPEncoder_SIjoint(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)

decoder_SI_joint = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
encoder_SI_joint.cuda()
decoder_SI_joint.cuda()
checkpoint = torch.load("SIprior_model/SIprior_best_model.pth", map_location=device)
emb_SI_prior.load_state_dict(checkpoint['emb_SI_prior_state_dict'])
encoder_SI_prior.load_state_dict(checkpoint['encoder_SI_prior_state_dict'])

optimizer_SIjoint = optim.Adam(list(encoder_SI_joint.parameters()) + list(decoder_SI_joint.parameters()),
                       lr=args.lr)
scheduler_SIjoint = lr_scheduler.StepLR(optimizer_SIjoint, step_size=args.lr_decay,
                                gamma=args.gamma)
if args.save_folder:
    # exp_counter = 0
    # now = datetime.datetime.now()
    # timestamp = now.isoformat().replace(":", "-").split(".")[0]
    # if not os.path.exists(args.save_folder):
    #     os.mkdir(args.save_folder)
    # name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
    #            args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
    # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    # save_folder = './{}/NRI-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.encoder,
    #                                                   args.decoder, timestamp)
    save_folder = 'SIjoint_data/'
    # os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    res_folder = save_folder + 'results/'
    # os.mkdir(res_folder)
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
best_val_loss = np.inf
for epoch in range(args.SI_joint_epochs):

    best_epoch = 0
    t_epoch_start = time.time()
    val_loss = train_SIjoint(epoch, encoder_SI_joint,decoder_SI_joint,emb_SI_prior,encoder_SI_prior,scheduler_SIjoint,train_loader,optimizer_SIjoint,rel_rec,rel_send,valid_loader,encoder_file,decoder_file,log,best_val_loss,args)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - t_epoch_start
    if args.b_walltime:
        if epoch_end_time - t_begin < 171900 - epoch_time:
            continue
        else:
            break
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
test_SIjoint(encoder_SI_joint,decoder_SI_joint,emb_SI_prior,encoder_SI_prior,rel_rec,rel_send,test_loader,encoder_file,decoder_file,save_folder,log,args)
if log is not None:
    print(save_folder)
    log.close()

####################
for _ in range(1):
    print("Generating pseudo labels using SIjoint")
    make_pseudo_labels_iteratively(rel_rec,rel_send,train_loader,valid_loader,test_loader,args)
    print(args.data_path)
    ####################
    print("Testing pseudo labels accuracy")
    check_pseudo_acc(args)
    print("Training SIprior in the first round")
    if args.suffix == "springs":
        train_loader_SI_prior, valid_loader_SI_prior, test_loader_SI_prior, loc_max, loc_min, vel_max, vel_min = load_springs_data_pse(
            args)
    else:
        train_loader_SI_prior, valid_loader_SI_prior, test_loader_SI_prior, loc_max, loc_min, vel_max, vel_min = load_netsims_data_pse(args)

    emb_SI_prior = MLP_emb(n_in=args.timesteps*args.dims, n_hid=args.SI_prior_hidden_dim,n_out=args.SI_prior_hidden_dim,do_prob=0.0).to(device)
    encoder_SI_prior = MLPEncoderpre(args.timesteps * args.dims, args.SI_prior_hidden_dim,
                             args.edge_types,
                             do_prob=0.0, factor=True).to(device)
    optimizer_SIprior=optim.Adam(
            list(emb_SI_prior.parameters()) + list(encoder_SI_prior.parameters()) ,
            lr=args.SI_prior_lr
        , weight_decay=1e-5)
    best_val_acc = 0.0
    best_model_path = "SIprior_model/SIprior_best_model.pth"
    for epoch in range(1, args.SI_prior_epochs+ 1):
        print(f"Epoch {epoch}/{args.SI_prior_epochs}")
        train_loss = train_SIprior_model(emb_SI_prior, encoder_SI_prior, train_loader_SI_prior, optimizer_SIprior, device, rel_rec, rel_send)
        val_loss, val_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, valid_loader_SI_prior, device, rel_rec, rel_send)
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.4f}")
        val_loss2, val_acc2 = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, valid_loader, device, rel_rec, rel_send)
        print(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss2={val_loss2:.6f}, Val Acc2={val_acc2:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    'epoch': epoch,
                    'emb_SI_prior_state_dict': emb_SI_prior.state_dict(),
                    'encoder_SI_prior_state_dict': encoder_SI_prior.state_dict(),
                    'optimizer_state_dict': optimizer_SIprior.state_dict(),
                    'val_acc': val_acc,
                }, best_model_path
            )
            print(f"==> New best model saved at epoch {epoch} with Val Acc2 {val_acc:.4f}")
    test_loss, test_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, test_loader_SI_prior, device, rel_rec, rel_send)

    print(f"Final Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

    print("Loading best model for evaluation on original data")
    checkpoint = torch.load(best_model_path)
    emb_SI_prior.load_state_dict(checkpoint['emb_SI_prior_state_dict'])
    encoder_SI_prior.load_state_dict(checkpoint['encoder_SI_prior_state_dict'])
    optimizer_SIprior.load_state_dict(checkpoint['optimizer_state_dict'])
    test_loss, test_acc = evaluate_SIpriormodel(emb_SI_prior, encoder_SI_prior, test_loader, device, rel_rec, rel_send)
    print(f"Best Model Test Loss={test_loss:.6f}, Test Accuracy={test_acc:.4f}")

    print("Training SIjoint in the first round")

    encoder_SI_joint = MLPEncoder_SIjoint(args.timesteps * args.dims, args.encoder_hidden,
                             args.edge_types,
                             args.encoder_dropout, args.factor)

    decoder_SI_joint = MLPDecoder(n_in_node=args.dims,
                             edge_types=args.edge_types,
                             msg_hid=args.decoder_hidden,
                             msg_out=args.decoder_hidden,
                             n_hid=args.decoder_hidden,
                             do_prob=args.decoder_dropout,
                             skip_first=args.skip_first)
    encoder_SI_joint.cuda()
    decoder_SI_joint.cuda()
    checkpoint = torch.load("SIprior_model/SIprior_best_model.pth", map_location=device)
    emb_SI_prior.load_state_dict(checkpoint['emb_SI_prior_state_dict'])
    encoder_SI_prior.load_state_dict(checkpoint['encoder_SI_prior_state_dict'])

    optimizer_SIjoint = optim.Adam(list(encoder_SI_joint.parameters()) + list(decoder_SI_joint.parameters()),
                           lr=args.lr)
    scheduler_SIjoint = lr_scheduler.StepLR(optimizer_SIjoint, step_size=args.lr_decay,
                                    gamma=args.gamma)
    if args.save_folder:
        # exp_counter = 0
        # now = datetime.datetime.now()
        # timestamp = now.isoformat().replace(":", "-").split(".")[0]
        # if not os.path.exists(args.save_folder):
        #     os.mkdir(args.save_folder)
        # name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
        #            args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
        # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        # save_folder = './{}/NRI-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.encoder,
        #                                                   args.decoder, timestamp)
        save_folder = 'SIjoint_data/'
        # os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')
        res_folder = save_folder + 'results/'
        # os.mkdir(res_folder)
        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')
        pickle.dump({'args': args}, open(meta_file, "wb"))
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")
    best_val_loss = np.inf
    for epoch in range(args.SI_joint_epochs):

        best_epoch = 0
        t_epoch_start = time.time()
        val_loss = train_SIjoint(epoch, encoder_SI_joint,decoder_SI_joint,emb_SI_prior,encoder_SI_prior,scheduler_SIjoint,train_loader,optimizer_SIjoint,rel_rec,rel_send,valid_loader,encoder_file,decoder_file,log,best_val_loss,args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - t_epoch_start
        if args.b_walltime:
            if epoch_end_time - t_begin < 171900 - epoch_time:
                continue
            else:
                break
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()
    test_SIjoint(encoder_SI_joint,decoder_SI_joint,emb_SI_prior,encoder_SI_prior,rel_rec,rel_send,test_loader,encoder_file,decoder_file,save_folder,log,args)
    if log is not None:
        print(save_folder)

        log.close()
