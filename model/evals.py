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


def sequence_acc(sequence1, sequence2):
  """Find the best matching of two sequences of integers

  Args: two numpy sequences

  Returns:
    best matching accuracy
    e.g. if sequence1=[0,0,1,2,2], sequence2=[3,3,4,4,1], then accuracy=4/5=0.8
  """

  unique_id1, counts1 = np.unique(sequence1, return_counts=True)
  unique_id2, _ = np.unique(sequence2, return_counts=True)
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
  for _, seq1_idx in enumerate(idx1):
    if len(copy_unique_id2) == 0:
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
  accuracy = np.max((sequence_acc(true_labels, predict_labels),
                     sequence_acc(predict_labels, true_labels)))
  return accuracy, len(true_labels)
