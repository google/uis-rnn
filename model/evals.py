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

import numpy as np


def compute_approximate_sequence_accuracy(sequence1, sequence2):
  """Find the best matching of two sequences of integers.

  For example, if sequence1=[0,0,1,2,2], sequence2=[3,3,4,4,1], then accuracy
  will be 4/5=0.8. Notice that the two sequences are not exchangable, in other
  words, you may get different results if you switch the order of the two
  sequences.

  Args:
    sequence1: a list of integer - The first sequence to match.
    sequence2: a list of integer - The second sequence to match.

  Returns:
    best matching accuracy
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
