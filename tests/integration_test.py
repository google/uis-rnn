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
"""Integration tests."""
import random
import tempfile
import unittest

import numpy as np
import torch

import uisrnn


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
  if not isinstance(cluster_id, list) or not cluster_id:
    raise ValueError("cluster_id must be a non-empty list")
  result = label_to_center[cluster_id[0]]
  for label in cluster_id[1:]:
    result = np.vstack((result, label_to_center[label]))
  noises = np.random.rand(*result.shape) * sigma
  return result + noises


class TestIntegration(unittest.TestCase):
  """Integration test that covers training, testing, and evaluation."""

  def setUp(self):
    # fix random seeds for reproducing results
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

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
    train_sequences = [
        train_sequence[:100, :],
        train_sequence[100:300, :],
        train_sequence[300:600, :],
        train_sequence[600:, :]
    ]
    train_cluster_ids = [
        train_cluster_id[:100],
        train_cluster_id[100:300],
        train_cluster_id[300:600],
        train_cluster_id[600:]
    ]

    # generate testing data
    test_cluster_id = ['A'] * 10 + ['B'] * 20 + ['C'] * 30 + ['D'] * 40
    random.shuffle(test_cluster_id)
    test_sequence = _generate_random_sequence(
        test_cluster_id, label_to_center, sigma=0.01)

    # construct model
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.rnn_depth = 2
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 2
    model_args.verbosity = 3
    training_args.learning_rate = 0.01
    training_args.learning_rate_half_life = 50
    training_args.train_iteration = 200
    training_args.enforce_cluster_id_uniqueness = False
    inference_args.test_iteration = 2

    model = uisrnn.UISRNN(model_args)

    # run training, and save the model
    model.fit(train_sequences, train_cluster_ids, training_args)
    temp_file_path = tempfile.mktemp()
    model.save(temp_file_path)

    # run testing
    predicted_label = model.predict(test_sequence, inference_args)

    # run evaluation
    model.logger.print(
        3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(
        predicted_label, test_cluster_id)
    self.assertEqual(1.0, accuracy)

    # load new model
    loaded_model = uisrnn.UISRNN(model_args)
    loaded_model.load(temp_file_path)

    # run testing with loaded model
    predicted_label = loaded_model.predict(test_sequence, inference_args)

    # run evaluation with loaded model
    model.logger.print(
        3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(
        predicted_label, test_cluster_id)
    self.assertEqual(1.0, accuracy)

    # keep training from loaded model on a subset of training data
    transition_bias_1 = model.transition_bias
    training_args.learning_rate = 0.001
    training_args.train_iteration = 50
    model.fit(train_sequence[:100, :], train_cluster_id[:100], training_args)
    transition_bias_2 = model.transition_bias
    self.assertNotAlmostEqual(transition_bias_1, transition_bias_2)
    model.logger.print(
        3, 'Asserting transition_bias changed from {} to {}'.format(
            transition_bias_1, transition_bias_2))

    # run evaluation
    model.logger.print(
        3, 'Asserting the equivalence between'
        '\nGround truth: {}\nPredicted: {}'.format(
            test_cluster_id, predicted_label))
    accuracy = uisrnn.compute_sequence_match_accuracy(
        predicted_label, test_cluster_id)
    self.assertEqual(1.0, accuracy)


if __name__ == '__main__':
  unittest.main()
