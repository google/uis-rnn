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

import random
import unittest

from model import arguments
from model import evals
from model import uisrnn
import numpy as np

def _generate_random_sequence(cluster_id, label_to_center, sigma=0.1):
  """A helper function to generate sequence.

  Args:
    cluster_id: a list of labels
    label_to_center: a dict from label to cluster center, where each center
      is a 1-d numpy array
    sigma: standard deviation of noise to be added to sequence

  Returns:
    a 2-d numpy array, with shape (length, observation_dim)
  """
  if not isinstance(cluster_id, list) or len(cluster_id) < 1:
    raise ValueError("cluster_id must be a non-empty list")
  result = label_to_center[cluster_id[0]]
  for id in cluster_id[1:]:
    result = np.vstack((result, label_to_center[id]))
  noises = np.random.rand(*result.shape) * sigma
  return result + noises


class TestIntegration(unittest.TestCase):

  def test_four_clusters(self):
    """Four clusters on vertices of a square."""
    label_to_center = {
      'A': np.array([0.0, 0.0]),
      'B': np.array([0.0, 1.0]),
      'C': np.array([1.0, 0.0]),
      'D': np.array([1.0, 1.0]),
    }

    # generate training data
    train_cluster_id = ['A'] * 400 + ['B'] * 300 + ['C'] * 200 + ['D'] * 100
    random.shuffle(train_cluster_id)
    train_sequence = _generate_random_sequence(
        train_cluster_id, label_to_center, sigma=0.01)

    # generate testing data
    test_cluster_id = ['A'] * 10 + ['B'] * 20 + ['C'] * 30 + ['D'] * 40
    random.shuffle(test_cluster_id)
    test_sequence = _generate_random_sequence(
        test_cluster_id, label_to_center, sigma=0.01)

    # construct model
    args = arguments.parse_arguments()
    args.rnn_depth = 2
    args.rnn_hidden_size = 8
    args.learning_rate = 0.01
    args.train_iteration = 200
    args.observation_dim = 2
    args.test_iteration = 2

    model = uisrnn.UISRNN(args)

    # training
    model.fit(args, train_sequence, np.array(train_cluster_id))

    # testing
    predicted_label = model.predict(args, test_sequence)

    # evaluation
    accuracy, length = evals.evaluate_result(predicted_label, test_cluster_id)
    self.assertEqual(1.0, accuracy)
    self.assertEqual(length, len(test_cluster_id))


if __name__ == '__main__':
  unittest.main()
