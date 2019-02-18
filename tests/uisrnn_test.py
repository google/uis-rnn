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
"""Tests for uisrnn.py."""
import tempfile
import unittest

import numpy as np

import uisrnn


class TestUISRNN(unittest.TestCase):
  """Test the UISRNN class."""

  def test_fit_concatenated_and_predict_single_label(self):
    """Train and test model while training data has single label.

    Training data have already been concatenated.
    """
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 16
    training_args.learning_rate = 0.01
    training_args.train_iteration = 50
    inference_args.test_iteration = 1

    # generate fake training data, assume already concatenated
    train_sequence = np.random.rand(1000, model_args.observation_dim)
    train_cluster_id = np.array(['A'] * 1000)

    model = uisrnn.UISRNN(model_args)

    # training
    model.fit(train_sequence, train_cluster_id, training_args)

    # testing, where data has less variation than training
    test_sequence = np.random.rand(10, model_args.observation_dim) / 10.0
    predicted_label = model.predict(test_sequence, inference_args)
    self.assertListEqual([0] * 10, predicted_label)

    # testing on two sequences
    test_sequence1 = np.random.rand(10, model_args.observation_dim) / 10.0
    test_sequence2 = np.random.rand(10, model_args.observation_dim) / 10.0
    predicted_cluster_ids = model.predict(
        [test_sequence1, test_sequence2], inference_args)
    self.assertIsInstance(predicted_cluster_ids, list)
    self.assertEqual(2, len(predicted_cluster_ids))
    self.assertListEqual([0] * 10, predicted_cluster_ids[0])
    self.assertListEqual([0] * 10, predicted_cluster_ids[1])

  def test_fit_list_and_predict_single_label(self):
    """Train and test model while training data has single label.

    Training data are not concatenated.
    """
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 16
    training_args.learning_rate = 0.01
    training_args.train_iteration = 50
    inference_args.test_iteration = 1

    # generate fake training data, as a list
    train_sequences = [
        np.random.rand(100, model_args.observation_dim),
        np.random.rand(200, model_args.observation_dim),
        np.random.rand(300, model_args.observation_dim)]
    train_cluster_ids = [
        np.array(['A'] * 100),
        np.array(['A'] * 200),
        np.array(['A'] * 300),]

    model = uisrnn.UISRNN(model_args)

    # training
    model.fit(train_sequences, train_cluster_ids, training_args)

    # testing, where data has less variation than training
    test_sequence = np.random.rand(10, model_args.observation_dim) / 10.0
    predicted_label = model.predict(test_sequence, inference_args)
    self.assertListEqual([0] * 10, predicted_label)

  def test_fit_with_wrong_dim(self):
    """Training data has wrong dimension."""
    model_args, training_args, _ = uisrnn.parse_arguments()
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 16
    training_args.learning_rate = 0.01
    training_args.train_iteration = 5

    # generate fake data
    train_sequence = np.random.rand(1000, 18)
    train_cluster_id = np.array(['A'] * 1000)

    model = uisrnn.UISRNN(model_args)

    # training
    with self.assertRaises(ValueError):
      model.fit(train_sequence, train_cluster_id, training_args)

  def test_predict_with_wrong_dim(self):
    """Testing data has wrong dimension."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 8
    model_args.observation_dim = 16
    training_args.learning_rate = 0.01
    training_args.train_iteration = 50

    # generate fake data
    train_sequence = np.random.rand(1000, model_args.observation_dim)
    train_cluster_id = np.array(['A'] * 1000)

    model = uisrnn.UISRNN(model_args)

    # training
    model.fit(train_sequence, train_cluster_id, training_args)

    # testing
    test_sequence = np.random.rand(10, 18)
    with self.assertRaises(ValueError):
      model.predict(test_sequence, inference_args)

  def test_save_and_load(self):
    """Save model and load it."""
    model_args, _, _ = uisrnn.parse_arguments()
    model_args.observation_dim = 16
    model_args.transition_bias = 0.5
    model_args.sigma2 = 0.05
    model = uisrnn.UISRNN(model_args)
    temp_file_path = tempfile.mktemp()
    model.save(temp_file_path)
    model.load(temp_file_path)
    self.assertEqual(0.5, model.transition_bias)


if __name__ == '__main__':
  unittest.main()
