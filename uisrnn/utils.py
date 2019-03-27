# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for UIS-RNN."""

import random
import string

import numpy as np
import torch
from torch import autograd


class Logger:
  """A class for printing logging information to screen."""

  def __init__(self, verbosity):
    self._verbosity = verbosity

  def print(self, level, message):
    """Print a message if level is not higher than verbosity.

    Args:
      level: the level of this message, smaller value means more important
      message: the message to be printed
    """
    if level <= self._verbosity:
      print(message)


def generate_random_string(length=6):
  """Generate a random string of upper case letters and digits.

  Args:
    length: length of the generated string

  Returns:
    the generated string
  """
  return ''.join([
      random.choice(string.ascii_uppercase + string.digits)
      for _ in range(length)])


def enforce_cluster_id_uniqueness(cluster_ids):
  """Enforce uniqueness of cluster id across sequences.

  Args:
    cluster_ids: a list of 1-dim list/numpy.ndarray of strings

  Returns:
    a new list with same length of cluster_ids

  Raises:
    TypeError: if cluster_ids or its element has wrong type
  """
  if not isinstance(cluster_ids, list):
    raise TypeError('cluster_ids must be a list')
  new_cluster_ids = []
  for cluster_id in cluster_ids:
    sequence_id = generate_random_string()
    if isinstance(cluster_id, np.ndarray):
      cluster_id = cluster_id.tolist()
    if not isinstance(cluster_id, list):
      raise TypeError('Elements of cluster_ids must be list or numpy.ndarray')
    new_cluster_id = ['_'.join([sequence_id, s]) for s in cluster_id]
    new_cluster_ids.append(new_cluster_id)
  return new_cluster_ids


def concatenate_training_data(train_sequences, train_cluster_ids,
                              enforce_uniqueness=True, shuffle=True):
  """Concatenate training data.

  Args:
    train_sequences: a list of 2-dim numpy arrays to be concatenated
    train_cluster_ids: a list of 1-dim list/numpy.ndarray of strings
    enforce_uniqueness: a boolean indicated whether we should enfore uniqueness
      to train_cluster_ids
    shuffle: whether to randomly shuffle input order

  Returns:
    concatenated_train_sequence: a 2-dim numpy array
    concatenated_train_cluster_id: a list of strings

  Raises:
    TypeError: if input has wrong type
    ValueError: if sizes/dimensions of input or their elements are incorrect
  """
  # check input
  if not isinstance(train_sequences, list) or not isinstance(
      train_cluster_ids, list):
    raise TypeError('train_sequences and train_cluster_ids must be lists')
  if len(train_sequences) != len(train_cluster_ids):
    raise ValueError(
        'train_sequences and train_cluster_ids must have same size')
  train_cluster_ids = [
      x.tolist() if isinstance(x, np.ndarray) else x
      for x in train_cluster_ids]
  global_observation_dim = None
  for i, (train_sequence, train_cluster_id) in enumerate(
      zip(train_sequences, train_cluster_ids)):
    train_length, observation_dim = train_sequence.shape
    if i == 0:
      global_observation_dim = observation_dim
    elif global_observation_dim != observation_dim:
      raise ValueError(
          'train_sequences must have consistent observation dimension')
    if not isinstance(train_cluster_id, list):
      raise TypeError(
          'Elements of train_cluster_ids must be list or numpy.ndarray')
    if len(train_cluster_id) != train_length:
      raise ValueError(
          'Each train_sequence and its train_cluster_id must have same length')

  # enforce uniqueness
  if enforce_uniqueness:
    train_cluster_ids = enforce_cluster_id_uniqueness(train_cluster_ids)

  # random shuffle
  if shuffle:
    zipped_input = list(zip(train_sequences, train_cluster_ids))
    random.shuffle(zipped_input)
    train_sequences, train_cluster_ids = zip(*zipped_input)

  # concatenate
  concatenated_train_sequence = np.concatenate(train_sequences, axis=0)
  concatenated_train_cluster_id = [x for train_cluster_id in train_cluster_ids
                                   for x in train_cluster_id]
  return concatenated_train_sequence, concatenated_train_cluster_id


def sample_permuted_segments(index_sequence, number_samples):
  """Sample sequences with permuted blocks.

  Args:
    index_sequence: (integer array, size: L)
      - subsequence index
      For example, index_sequence = [1,2,6,10,11,12].
    number_samples: (integer)
      - number of subsampled block-preserving permuted sequences.
      For example, number_samples = 5

  Returns:
    sampled_index_sequences: (a list of numpy arrays) - a list of subsampled
      block-preserving permuted sequences. For example,
    ```
    sampled_index_sequences =
    [[10,11,12,1,2,6],
     [6,1,2,10,11,12],
     [1,2,10,11,12,6],
     [6,1,2,10,11,12],
     [1,2,6,10,11,12]]
    ```
      The length of "sampled_index_sequences" is "number_samples".
  """
  segments = []
  if len(index_sequence) == 1:
    segments.append(index_sequence)
  else:
    prev = 0
    for i in range(len(index_sequence) - 1):
      if index_sequence[i + 1] != index_sequence[i] + 1:
        segments.append(index_sequence[prev:(i + 1)])
        prev = i + 1
      if i + 1 == len(index_sequence) - 1:
        segments.append(index_sequence[prev:])
  # sample permutations
  sampled_index_sequences = []
  for _ in range(number_samples):
    segments_array = []
    permutation = np.random.permutation(len(segments))
    for permutation_item in permutation:
      segments_array.append(segments[permutation_item])
    sampled_index_sequences.append(np.concatenate(segments_array))
  return sampled_index_sequences


def resize_sequence(sequence, cluster_id, num_permutations=None):
  """Resize sequences for packing and batching.

  Args:
    sequence: (real numpy matrix, size: seq_len*obs_size) - observed sequence
    cluster_id: (numpy vector, size: seq_len) - cluster indicator sequence
    num_permutations: int - Number of permutations per utterance sampled.

  Returns:
    sub_sequences: A list of numpy array, with obsevation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
    bias: Flipping coin head probability.
    bias_denominator: The denominator of the bias, used for multiple calls to
      fit().
  """
  # merge sub-sequences that belong to a single cluster to a single sequence
  unique_id = np.unique(cluster_id)
  sub_sequences = []
  seq_lengths = []
  if num_permutations and num_permutations > 1:
    for i in unique_id:
      idx_set = np.where(cluster_id == i)[0]
      sampled_idx_sets = sample_permuted_segments(idx_set, num_permutations)
      for j in range(num_permutations):
        sub_sequences.append(sequence[sampled_idx_sets[j], :])
        seq_lengths.append(len(idx_set) + 1)
  else:
    for i in unique_id:
      idx_set = np.where(cluster_id == i)
      sub_sequences.append(sequence[idx_set, :][0])
      seq_lengths.append(len(idx_set[0]) + 1)

  # compute bias
  transit_num = 0
  for entry in range(len(cluster_id) - 1):
    transit_num += (cluster_id[entry] != cluster_id[entry + 1])
  bias_denominator = len(cluster_id)
  bias = (transit_num + 1) / bias_denominator
  return sub_sequences, seq_lengths, bias, bias_denominator


def pack_sequence(
    sub_sequences, seq_lengths, batch_size, observation_dim, device):
  """Pack sequences for training.

  Args:
    sub_sequences: A list of numpy array, with obsevation vector from the same
      cluster in the same list.
    seq_lengths: The length of each cluster (+1).
    batch_size: int or None - Run batch learning if batch_size is None. Else,
      run online learning with specified batch size.
    observation_dim: int - dimension for observation vectors
    device: str - Your device. E.g., `cuda:0` or `cpu`.

  Returns:
    packed_rnn_input: (PackedSequence object) packed rnn input
    rnn_truth: ground truth
  """
  num_clusters = len(seq_lengths)
  sorted_seq_lengths = np.sort(seq_lengths)[::-1]
  permute_index = np.argsort(seq_lengths)[::-1]

  if batch_size is None:
    rnn_input = np.zeros((sorted_seq_lengths[0],
                          num_clusters,
                          observation_dim))
    for i in range(num_clusters):
      rnn_input[1:sorted_seq_lengths[i], i,
                :] = sub_sequences[permute_index[i]]
    rnn_input = autograd.Variable(
        torch.from_numpy(rnn_input).float()).to(device)
    packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
        rnn_input, sorted_seq_lengths, batch_first=False)
  else:
    mini_batch = np.sort(np.random.choice(num_clusters, batch_size))
    rnn_input = np.zeros((sorted_seq_lengths[mini_batch[0]],
                          batch_size,
                          observation_dim))
    for i in range(batch_size):
      rnn_input[1:sorted_seq_lengths[mini_batch[i]],
                i, :] = sub_sequences[permute_index[mini_batch[i]]]
    rnn_input = autograd.Variable(
        torch.from_numpy(rnn_input).float()).to(device)
    packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
        rnn_input, sorted_seq_lengths[mini_batch], batch_first=False)
  # ground truth is the shifted input
  rnn_truth = rnn_input[1:, :, :]
  return packed_rnn_input, rnn_truth


def output_result(model_args, training_args, test_record):
  """Produce a string to summarize the experiment."""
  accuracy_array, _ = zip(*test_record)
  total_accuracy = np.mean(accuracy_array)
  output_string = """
Config:
  sigma_alpha: {}
  sigma_beta: {}
  crp_alpha: {}
  learning rate: {}
  regularization: {}
  batch size: {}

Performance:
  averaged accuracy: {:.6f}
  accuracy numbers for all testing sequences:
  """.strip().format(
      training_args.sigma_alpha,
      training_args.sigma_beta,
      model_args.crp_alpha,
      training_args.learning_rate,
      training_args.regularization_weight,
      training_args.batch_size,
      total_accuracy)
  for accuracy in accuracy_array:
    output_string += '\n    {:.6f}'.format(accuracy)
  output_string += '\n' + '=' * 80 + '\n'
  filename = 'layer_{}_{}_{:.1f}_result.txt'.format(
      model_args.rnn_hidden_size,
      model_args.rnn_depth, model_args.rnn_dropout)
  with open(filename, 'a') as file_object:
    file_object.write(output_string)
  return output_string
