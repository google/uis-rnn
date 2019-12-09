import numpy as np
from tqdm import tqdm

search_range = 100
search_step = 0.1


def get_alpha(train_cluster_id):
    cur_alpha, cur_map = np.nan, -np.inf
    for alpha in tqdm(range(1, search_range)):
        map = get_map(train_cluster_id, alpha * search_step)
    if map > cur_map:
        cur_alpha, cur_map = alpha, map
    return cur_alpha * search_step


def get_map(train_cluster_id, alpha):
    map = 0

    for id in get_id_single(train_cluster_id):
        map_single = np.log(get_map_single(id, alpha))
        map += map_single
    return map


def get_map_single(train_cluster_id, alpha):
    kt = get_kt(train_cluster_id)
    Nkt = get_nkt(train_cluster_id)
    numerator = alpha ** (len(set(train_cluster_id)) - 1)
    denominator = 1

    for i in range(1, len(train_cluster_id)):
        if train_cluster_id[i] != train_cluster_id[i - 1]:
            denominator_i = sum([Nkt[i - 1, j] for j in range(kt[i - 1]) if j != train_cluster_id[i - 1]]) + alpha
    denominator *= denominator_i
    map_single = numerator / denominator
    return map_single


def get_kt(train_cluster_id):
    kt = np.array([len(set(train_cluster_id[:i + 1])) for i in range(len(train_cluster_id))])

    return kt


def get_nkt(train_cluster_id):
    num_spk = len(set(train_cluster_id))
    nkt = np.zeros((len(train_cluster_id), num_spk))
    cur_nkt = np.zeros((num_spk))

    for i, j in enumerate(train_cluster_id):
        if i == 0:
            cur_spk = j
            cur_nkt[j] += 1
            continue
        if j != cur_spk:
            cur_spk = j
            cur_nkt[j] += 1
        nkt[i] = cur_nkt
    return nkt


def get_id_single(train_cluster_id):
    cur_index, cur_prefix = 0, train_cluster_id[0].split('_')[0]

    for index in range(len(train_cluster_id)):
        prefix = train_cluster_id[index].split('_')[0]
    if prefix != cur_prefix or index == len(train_cluster_id) - 1:
        yield get_normalized_id(train_cluster_id[cur_index: index])
        cur_index, cur_prefix = index, prefix


def get_normalized_id(train_cluster_id):
    normalized_id = [int(i.split('_')[1]) for i in train_cluster_id]
    index_order = [np.nan] * len(set(train_cluster_id))
    count = 0

    for i in normalized_id:
        if i not in index_order:
            index_order[count] = i
            count += 1
        if count == len(index_order):
            break
    normalized_id = np.array([index_order.index(i) for i in normalized_id])
    return normalized_id
