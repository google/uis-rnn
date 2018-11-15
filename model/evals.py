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
"""Utils for model evaluation."""

from scipy import optimize
import numpy as np


def get_list_inverse_index(unique_ids):
  """Get value to position index from a list of unique ids.

  Args:
    unique_ids: A list of unique integers of strings.

  Returns:
    a dict from value to position

  Raises:
    TypeError: If unique_ids is not a list.
  """
  if not isinstance(unique_ids, list):
    raise TypeError('unique_ids must be a list')
  result = dict()
  for i, id in enumerate(unique_ids):
    result[id] = i
  return result


def compute_sequence_match_accuracy(sequence1, sequence2):
  """Compute the accuracy between two sequences by finding optimal matching.

  Args:
    sequence1: A list of integers or strings.
    sequence2: A list of integers or strings.

  Returns:
    sequence matching accuracy as a number in [0.0, 1.0]

  Raises:
    TypeError: If sequence1 or sequence2 is not list.
    ValueError: If sequence1 and sequence2 are not same size.
  """
  if not isinstance(sequence1, list) or not isinstance(sequence2, list):
    raise TypeError('sequence1 and sequence2 must be lists')
  if not sequence1 or len(sequence1) != len(sequence2):
    raise ValueError(
        'sequence1 and sequence2 must have the same non-zero length')
  # get unique ids from sequences
  unique_ids1 = sorted(set(sequence1))
  unique_ids2 = sorted(set(sequence2))
  inverse_index1 = get_list_inverse_index(unique_ids1)
  inverse_index2 = get_list_inverse_index(unique_ids2)
  # get the count matrix
  count_matrix = np.zeros((len(unique_ids1), len(unique_ids2)))
  for i in range(len(sequence1)):
    index1 = inverse_index1[sequence1[i]]
    index2 = inverse_index2[sequence2[i]]
    count_matrix[index1, index2] += 1.0
  row_index, col_index = optimize.linear_sum_assignment(-count_matrix)
  optimal_match_count = count_matrix[row_index, col_index].sum()
  accuracy = optimal_match_count / len(sequence1)
  return accuracy


def compute_approximate_sequence_accuracy(sequence1, sequence2):
  """Find a sub-optimal matching of two sequences of integers.

  This sub-optimal matching is obtained by a greedy algorithm. It's not the
  best matching. To compute the real best matching, the Hungarian algorithm
  is needed, which requires additional dependencies.

  For example, if sequence1=[0,0,1,2,2], sequence2=[3,3,4,4,1], then accuracy
  will be 4/5=0.8. Notice that the two sequences are not exchangable, in other
  words, you may get different results if you switch the order of the two
  sequences.

  Args:
    sequence1: a list of integer - The first sequence to match.
    sequence2: a list of integer - The second sequence to match.

  Returns:
    sub-optimal matching accuracy
  """
  assert len(sequence1) == len(sequence2), (
      "The two sequences should should of the same length")
  assert sequence1, "The sequences cannot be empty"

  unique_id1, counts1 = np.unique(sequence1, return_counts=True)
  unique_id2, _ = np.unique(sequence2, return_counts=True)
  # transform into numpy arrays
  dict1 = dict(zip(unique_id1, np.arange(len(unique_id1))))
  dict2 = dict(zip(unique_id2, np.arange(len(unique_id2))))
  sequence1 = np.array([dict1[k] for k in sequence1])
  sequence2 = np.array([dict2[k] for k in sequence2])
  unique_id1 = np.arange(len(unique_id1))
  unique_id2 = np.arange(len(unique_id2))

  idx1 = np.argsort(counts1)[::-1]
  n_forward_match = 0
  copy_unique_id2 = np.copy(unique_id2)
  for seq1_idx in idx1:
    if not copy_unique_id2.shape[0]:
      break
    n_match = []
    for seq2_idx in copy_unique_id2:
      n_match.append(
          len(
              np.intersect1d(
                  np.where(sequence1 == unique_id1[seq1_idx])[0],
                  np.where(sequence2 == seq2_idx)[0])))
    best_seq2_idx = np.argmax(n_match)
    n_forward_match += np.max(n_match)
    copy_unique_id2 = np.delete(copy_unique_id2, best_seq2_idx)

  return n_forward_match / len(sequence1)


def evaluate_result(true_labels, predict_labels):
  accuracy = np.max((
      compute_approximate_sequence_accuracy(true_labels, predict_labels),
      compute_approximate_sequence_accuracy(predict_labels, true_labels)))
  return accuracy, len(true_labels)
