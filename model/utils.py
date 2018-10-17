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


def sequence_acc(sequence1, sequence2):
  '''Find the best matching of two sequences of integers

  Args:
    two numpy sequences
  Returns:
    best matching accuracy
    e.g. if sequence1=[0,0,1,2,2], sequence2=[3,3,4,4,1], then accuracy=3/5=0.6
  '''

  unique_id1, counts1 = np.unique(sequence1, return_counts=True)
  unique_id2, counts2 = np.unique(sequence2, return_counts=True)
  # transform into numpy arrays
  dict1 = dict(zip(unique_id1, np.arange(len(unique_id1))))
  dict2 = dict(zip(unique_id2, np.arange(len(unique_id2))))
  sequence1 = np.array([dict1[sequence1[i]] for i in range(len(sequence1))])
  sequence2 = np.array([dict2[sequence2[i]] for i in range(len(sequence2))])
  unique_id1 = np.arange(len(unique_id1))
  unique_id2 = np.arange(len(unique_id2))

  idx1 = np.argsort(counts1)[::-1]
  n_forward_match = 0
  copy_unique_id2 = np.copy(unique_id2)
  for seq1_rank, seq1_idx in enumerate(idx1):
    if len(copy_unique_id2) == 0:
      break
    n_match = []
    for seq2_idx in copy_unique_id2:
      n_match.append(len(
        np.intersect1d(np.where(sequence1==unique_id1[seq1_idx])[0], np.where(sequence2==seq2_idx)[0])))
    best_seq2_idx = np.argmax(n_match)
    n_forward_match += np.max(n_match)
    copy_unique_id2 = np.delete(copy_unique_id2, best_seq2_idx)

  return n_forward_match/len(sequence1)


def evaluate_result(args, true_labels, predict_labels):
  accuracy = np.max((sequence_acc(true_labels,predict_labels),sequence_acc(predict_labels,true_labels)))
  return accuracy, len(true_labels)


def output_result(args, test_record):
  accuracy_array, length_array = zip(*test_record)
  # print(accuracy_array, length_array)
  total_accuracy = np.mean(accuracy_array)
  file = open('{}_{}_layer{}_{}_{:.1f}_result.txt'.format(args.dataset, args.model_type, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout), 'a')
  file.write('dataset:{}  alpha:{}  beta:{}  crp_theta:{}  learning rate:{}  regularization:{}  batch size:{}  acc:{:.6f} \n'.format(
    args.dataset, args.alpha, args.beta, args.crp_theta, args.learn_rate, args.network_reg, args.batch_size, total_accuracy))

  file.close()
