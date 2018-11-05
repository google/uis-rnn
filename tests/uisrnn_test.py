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

import tempfile
import unittest

from model import arguments
from model import uisrnn
import numpy as np


class TestUISRNN(unittest.TestCase):

  def test_fit_and_predict(self):
    args = arguments.parse_arguments()
    args.train_iteration = 10
    args.observation_dim = 16
    args.rnn_hidden_size = 16

    # generate fake data
    train_sequence = np.random.rand(1000, args.observation_dim)
    train_cluster_id = np.array(['A'] * 1000)
    _, observation_dim = train_sequence.shape

    model = uisrnn.UISRNN(args, .5)

    # training
    model.fit(args, train_sequence, train_cluster_id)

    # testing
    test_sequence = np.random.rand(10, args.observation_dim)
    predicted_label = model.predict(args, test_sequence)

  def test_fit_with_wrong_dim(self):
    args = arguments.parse_arguments()
    args.train_iteration = 10
    args.observation_dim = 16
    args.rnn_hidden_size = 16

    # generate fake data
    train_sequence = np.random.rand(1000, 18)
    train_cluster_id = np.array(['A'] * 1000)
    _, observation_dim = train_sequence.shape

    model = uisrnn.UISRNN(args, .5)

    # training
    with self.assertRaises(ValueError):
      model.fit(args, train_sequence, train_cluster_id)

  def test_predict_with_wrong_dim(self):
    args = arguments.parse_arguments()
    args.train_iteration = 10
    args.observation_dim = 16
    args.rnn_hidden_size = 16

    # generate fake data
    train_sequence = np.random.rand(1000, args.observation_dim)
    train_cluster_id = np.array(['A'] * 1000)
    _, observation_dim = train_sequence.shape

    model = uisrnn.UISRNN(args, .5)

    # training
    model.fit(args, train_sequence, train_cluster_id)

    # testing
    test_sequence = np.random.rand(10, 18)
    with self.assertRaises(ValueError):
      predicted_label = model.predict(args, test_sequence)

  def test_save_and_load(self):
    args = arguments.parse_arguments()
    args.observation_dim = 16
    model = uisrnn.UISRNN(args, .5)
    temp_file_path = tempfile.mktemp()
    model.save(temp_file_path)
    model.load(temp_file_path)


if __name__ == '__main__':
  unittest.main()
