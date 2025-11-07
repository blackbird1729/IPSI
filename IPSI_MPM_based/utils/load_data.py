import pickle
import numpy as np
def read_pickle(infile):
    """
    A wrapper for pickle.load().
    """
    with open(infile, 'rb') as f:
        return pickle.load(f)

def load_data(args):
    path = 'data/{}/{}.pkl'.format(args.dyn, args.size)
    train, val, test = read_pickle(path)
    data = {'train': train, 'val': val, 'test': test}
    return data


def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]


def load_customized_data(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    if args.dyn == 'springs':
        train, val, test = load_customized_springs_data(args, keep_str, root_str)
    elif args.dyn == 'netsims':
        train, val, test = load_customized_netsims_data(args, keep_str, root_str)
    else:
        raise ValueError("Check args.dyn!")
    data = {'train': train, 'val': val, 'test': test}
    return data


def load_customized_springs_data(args, keep_str, root_str):
    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_test = portion_data(loc_test, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_test = portion_data(vel_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    loc_train = np.transpose(loc_train, [0, 1, 3, 2])
    vel_train = np.transpose(vel_train, [0, 1, 3, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    loc_valid = np.transpose(loc_valid, [0, 1, 3, 2])
    vel_valid = np.transpose(vel_valid, [0, 1, 3, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    loc_test = np.transpose(loc_test, [0, 1, 3, 2])
    vel_test = np.transpose(vel_test, [0, 1, 3, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test


def load_customized_netsims_data(args, keep_str, root_str):
    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_test = portion_data(bold_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = bold_train.shape[3]

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 1, 3, 2])
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    feat_valid = np.transpose(bold_valid, [0, 1, 3, 2])
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    feat_test = np.transpose(bold_test, [0, 1, 3, 2])
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test

def load_customized_data_pse(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    if args.dyn == 'springs':
        train, val, test = load_customized_springs_data_pse(args, keep_str, root_str)
    elif args.dyn == 'netsims':
        train, val, test = load_customized_netsims_data_pse(args, keep_str, root_str)
    else:
        raise ValueError("Check args.dyn!")
    data = {'train': train, 'val': val, 'test': test}
    return data


def load_customized_springs_data_pse(args, keep_str, root_str):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load('pseudo_data/' + 'edges_train_pseudo.npy')
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load('pseudo_data/' + 'edges_val_pseudo.npy')
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load('pseudo_data/' + 'edges_test_pseudo.npy')
    edges_test[edges_test > 0] = 1

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_test = portion_data(loc_test, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_test = portion_data(vel_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    loc_train = np.transpose(loc_train, [0, 1, 3, 2])
    vel_train = np.transpose(vel_train, [0, 1, 3, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    loc_valid = np.transpose(loc_valid, [0, 1, 3, 2])
    vel_valid = np.transpose(vel_valid, [0, 1, 3, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    loc_test = np.transpose(loc_test, [0, 1, 3, 2])
    vel_test = np.transpose(vel_test, [0, 1, 3, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test

def load_customized_netsims_data(args, keep_str, root_str):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'
    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load('pseudo_data/' + 'edges_train_pseudo.npy')
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load('pseudo_data/' + 'edges_valid_pseudo.npy')
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load('pseudo_data/' + 'edges_test_pseudo.npy')
    edges_test[edges_test > 0] = 1

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_test = portion_data(bold_test, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = bold_train.shape[3]

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 1, 3, 2])
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    feat_valid = np.transpose(bold_valid, [0, 1, 3, 2])
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    feat_test = np.transpose(bold_test, [0, 1, 3, 2])
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    train = list()
    val = list()
    test = list()

    for i in range(n_train):
        train.append((edges_train[i], feat_train[i]))
    for i in range(n_valid):
        val.append((edges_valid[i], feat_valid[i]))
    for i in range(n_test):
        test.append((edges_test[i], feat_test[i]))
    return train, val, test