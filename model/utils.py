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

import numpy as np
import torch


def weighted_mse_loss(input, target, weight=1):
  '''Compute weighted mse loss

  Note that we are doing weighted loss that only sum up over non-zero entries.

  Args:
    input: input tensor
    target: target tensor
    weight: weight tensor, in this case 1/sigma^2
  Returns:
    weighted mse loss
  '''
  observation_dim = input.size()[-1]
  streched_tensor = ((input-target)**2).view(-1,observation_dim)
  entry_num = float(streched_tensor.size()[0])
  non_zero_entry_num = torch.sum(streched_tensor[:,0]!=0).float()
  weighted_tensor = torch.mm(((input-target)**2).view(-1,observation_dim),(torch.diag(weight.float().view(-1))))
  return torch.mean(weighted_tensor)*weight.nelement()*entry_num/non_zero_entry_num


def sample_permuted_segments(index_sequence, number_samples):
  ''' To be added
  '''
  segments = []
  if len(index_sequence) == 1:
    segments.append(index_sequence)
  else:
    prev = 0
    for i in range(len(index_sequence)-1):
      if (index_sequence[i+1] != index_sequence[i]+1):
        segments.append(index_sequence[prev:(i+1)])
        prev = i+1
      if i+1 == len(index_sequence)-1:
        segments.append(index_sequence[prev:])
  # sample permutations
  sampled_index_sequences = []
  for n in range(number_samples):
    segments_array = []
    permutation = np.random.permutation(len(segments))
    for i in range(len(permutation)):
      segments_array.append(segments[permutation[i]])
    sampled_index_sequences.append(np.concatenate(segments_array))
  return sampled_index_sequences


def resize_seq(args, sequence, cluster_id):
  '''Resize sequences for packing and batching

  Args:
    sequence (real numpy matrix, size: seq_len*obs_size): observation sequence
    cluster_id (real vector, size: seq_len): cluster indicator sequence
  Returns:
    packed_rnn_input:
    rnn_truth:
    bias: flipping coin head probability
  '''

  obs_size = np.shape(sequence)[1]
  # merge sub-sequences that belong to a single cluster to a single sequence
  unique_id = np.unique(cluster_id)
  if args.permutation is None:
    num_clusters = len(unique_id)
  else:
    num_clusters = len(unique_id)*args.permutation
  sub_sequences = []
  seq_lengths = []
  if args.permutation is None:
    for i in unique_id:
      sub_sequences.append(sequence[np.where(cluster_id==i),:][0])
      seq_lengths.append(len(np.where(cluster_id==i)[0])+1)
  else:
    for i in unique_id:
      idx_set = np.where(cluster_id==i)[0]
      sampled_idx_sets = sample_permuted_segments(idx_set, args.permutation)
      for j in range(args.permutation):
        sub_sequences.append(sequence[sampled_idx_sets[j],:])
        seq_lengths.append(len(idx_set)+1)

  # compute bias
  transit_num = 0
  for entry in range(len(cluster_id)-1):
     transit_num += (cluster_id[entry]!=cluster_id[entry+1])
  # return sub_sequences, seq_lengths, transit_num/(len(cluster_id)-1)
  return sub_sequences, seq_lengths, 0.158


def pack_seq(rnn_input, sorted_seq_lengths):
  packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_seq_lengths, batch_first=False)
  # ground truth is the shifted input
  rnn_truth = rnn_input[1:,:,:]
  return packed_rnn_input, rnn_truth


def output_result(args, test_record):
  accuracy_array, length_array = zip(*test_record)
  # print(accuracy_array, length_array)
  total_accuracy = np.mean(accuracy_array)
  file = open('{}_{}_layer{}_{}_{:.1f}_result.txt'.format(args.dataset, args.model_type, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout), 'a')
  file.write('dataset:{}  alpha:{}  beta:{}  crp_theta:{}  learning rate:{}  regularization:{}  batch size:{}  acc:{:.6f} \n'.format(
    args.dataset, args.alpha, args.beta, args.crp_theta, args.learn_rate, args.network_reg, args.batch_size, total_accuracy))

  file.close()
