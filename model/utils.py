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

import numpy as np
import torch


def weighted_mse_loss(input_tensor, target_tensor, weight=1):
  """Compute weighted mse loss.

  Note that we are doing weighted loss that only sum up over non-zero entries.

  Args:
    input_tensor: input tensor
    target_tensor: target tensor
    weight: weight tensor, in this case 1/sigma^2

  Returns:
    weighted mse loss
  """
  observation_dim = input_tensor.size()[-1]
  streched_tensor = ((input_tensor - target_tensor)**2).view(-1,
                                                             observation_dim)
  entry_num = float(streched_tensor.size()[0])
  non_zero_entry_num = torch.sum(streched_tensor[:, 0] != 0).float()
  weighted_tensor = torch.mm(
      ((input_tensor - target_tensor)**2).view(-1, observation_dim),
      (torch.diag(weight.float().view(-1))))
  return torch.mean(
      weighted_tensor) * weight.nelement() * entry_num / non_zero_entry_num


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
    sampled_index_sequences: (a list of numpy arrays)
      - a list of subsampled block-preserving permuted sequences.
      For example, sampled_index_sequences =
      [[10,11,12,1,2,6],
       [6,1,2,10,11,12],
       [1,2,10,11,12,6],
       [6,1,2,10,11,12],
       [1,2,6,10,11,12]]
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
    for i in range(len(permutation)):
      segments_array.append(segments[permutation[i]])
    sampled_index_sequences.append(np.concatenate(segments_array))
  return sampled_index_sequences


def resize_sequence(sequence, cluster_id, num_permutations=None):
  """Resize sequences for packing and batching.

  Args:
    sequence: (real numpy matrix, size: seq_len*obs_size) - observed sequence
    cluster_id: (numpy vector, size: seq_len) - cluster indicator sequence
    num_permutations: int - Number of permutations per utterance sampled.

  Returns:
    sub_sequences: a list of numpy array, with obsevation vector from the same
      cluster in the same list.
    seq_lengths: the length of each cluster (+1)
    bias: flipping coin head probability.
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
  bias = (transit_num + 1) / len(cluster_id)
  return sub_sequences, seq_lengths, bias


def pack_seq(rnn_input, sorted_seq_lengths):
  packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(
      rnn_input, sorted_seq_lengths, batch_first=False)
  # ground truth is the shifted input
  rnn_truth = rnn_input[1:, :, :]
  return packed_rnn_input, rnn_truth


def output_result(args, test_record):
  accuracy_array, _ = zip(*test_record)
  total_accuracy = np.mean(accuracy_array)
  filename = 'layer{}_{}_{:.1f}_result.txt'.format(
      args.rnn_hidden_size,
      args.rnn_depth, args.rnn_dropout)
  with open(filename, 'a') as file:
    file.write(
        'sigma_alpha:{}  sigma_beta:{}  crp_alpha:{}  learning rate:{}  '
        'regularization:{}  batch size:{}  acc:{:.6f} \n'
        .format(args.sigma_alpha, args.sigma_beta, args.crp_alpha,
                args.learning_rate, args.regularization_weight,
                args.batch_size, total_accuracy))
