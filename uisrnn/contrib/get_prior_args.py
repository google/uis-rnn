# Contributor information:
#   Name: Xiang Lyu
#   GitHub account: aluminumbox
#   Email: aluminumbox@alumni.sjtu.edu.cn
#   Organization: Ping An Technology (Shanghai) Co., Ltd.
import numpy as np

SEARCH_RANGE = 100
SEARCH_STEP = 0.01


def get_alpha(train_cluster_id):
  """Iterate through a range of alpha, return alpha with maximum cdf P{Y|Z}.

  Args:
    train_cluster_id: same as train_cluster_id in demo.py. See `demo.py` for
      details.
  Returns:
    cur_alpha: a float variable.
  """
  cur_alpha, cur_cdf = np.nan, -np.inf
  for alpha in range(1, SEARCH_RANGE):
    cdf = get_cdf(train_cluster_id, alpha * SEARCH_STEP)
    if cdf > cur_cdf:
      cur_alpha, cur_cdf = alpha * SEARCH_STEP, cdf
  return cur_alpha


def get_cdf(train_cluster_id, alpha):
  """For a given alpha, calculate the cdf of the entire observation sequence.

  Args:
    train_cluster_id: same as train_cluster_id in demo.py. See `demo.py` for
      details.
    alpha: a float variable.
  Returns:
    cdf: cdf of the entire observation sequence.
  """
  cdf = 0
  for cluster_id_single in get_cluster_id_single(train_cluster_id):
    cdf_single = np.log(get_cdf_single(cluster_id_single, alpha))
    cdf += cdf_single
  return cdf


def get_cdf_single(cluster_id_single, alpha):
  """For a given alpha, calculate the cdf of a single observation sequence.

  Args:
    cluster_id_single: train_cluster_id of a single observation sequence.
    alpha: a float variable.
  Returns:
    cdf_single: cdf of a single observation sequence.
  """
  k_t = get_k_t(cluster_id_single)
  n_kt = get_n_kt(cluster_id_single)
  numerator = alpha ** (len(set(cluster_id_single)) - 1)
  denominator = 1
  for i in range(1, len(cluster_id_single)):
    if cluster_id_single[i] != cluster_id_single[i - 1]:
      denominator_i = sum([n_kt[i - 1, j] for j in range(k_t[i - 1])
                           if j != cluster_id_single[i - 1]]) + alpha
      denominator *= denominator_i
  cdf_single = numerator / denominator
  return cdf_single


def get_k_t(cluster_id_single):
  """For a single observation sequence, calculate K_t. See Eq.8 in paper.

  Args:
    cluster_id_single: train_cluster_id of a single observation sequence.
  Returns:
    k_t: a numpy array.
  """
  k_t = np.array([len(set(cluster_id_single[:i + 1])) for i in
                  range(len(cluster_id_single))])
  return k_t


def get_n_kt(cluster_id_single):
  """For a given observation sequence, calculate N_{k,t}. See Eq.8 in paper.

  Args:
    cluster_id_single: train_cluster_id of a single observation sequence.
  Returns:
    n_kt: a numpy array.
  """
  num_spk = len(set(cluster_id_single))
  n_kt = np.zeros((len(cluster_id_single), num_spk))
  cur_n_kt = np.zeros((num_spk))
  for i, j in enumerate(cluster_id_single):
    if i == 0:
      cur_spk = j
      cur_n_kt[j] += 1
      continue
    if j != cur_spk:
      cur_spk = j
      cur_n_kt[j] += 1
    n_kt[i] = cur_n_kt
  return n_kt


def get_cluster_id_single(train_cluster_id):
  """Given the entire observation sequence, yields normalized id for a single
  observation sequence each time

  Args:
    train_cluster_id: same as train_cluster_id in demo.py. See `demo.py` for
      details.
  Yields:
    cluster_id_single: normalized id for a single observation sequence.
  For example:
  ```
    train_cluster_id = [0_0, 0_0, 0_2, 0_2, 0_1, 0_1, 1_0, 1_1, 1_1, 1_2]
    yields [0, 0, 1, 1, 2, 2], [0, 1, 1, 2]
  ```
  """
  cur_index, cur_prefix = 0, train_cluster_id[0].split('_')[0]
  for i, j in enumerate(train_cluster_id):
    prefix = j.split('_')[0]
    if prefix != cur_prefix or i == len(train_cluster_id) - 1:
      cluster_id_single = get_normalized_id(train_cluster_id[cur_index: i])
      yield cluster_id_single
      cur_index, cur_prefix = i, prefix


def get_normalized_id(cluster_id_single):
  """For a single observation sequence, returns its normalized form.

  Args:
    cluster_id_single: train_cluster_id for a single observation sequence.
  Returns:
    normalized_id: normalized id for a single observation sequence.
  For example:
  ```
    train_cluster_id = [0_0, 0_0, 0_2, 0_2, 0_1, 0_1]
    normalized_id = [0, 0, 1, 1, 2, 2]
  ```
  """
  normalized_id = [int(i.split('_')[1]) for i in cluster_id_single]
  index_order = [np.nan] * len(set(cluster_id_single))
  count = 0
  for i in normalized_id:
    if i not in index_order:
      index_order[count] = i
      count += 1
    if count == len(index_order):
      break
  normalized_id = np.array([index_order.index(i) for i in normalized_id])
  return normalized_id
