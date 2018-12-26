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
"""Tests for utils.py."""
import unittest

import numpy as np

from uisrnn import utils


class TestSamplePermutedSegments(unittest.TestCase):
  """Tests for utils.sample_permuted_segments()"""

  def test_short_sequence(self):
    """Test for a short sequence."""
    index_sequence = [5, 2, 3, 2, 1]
    number_samples = 10
    sampled_index_sequences = utils.sample_permuted_segments(index_sequence,
                                                             number_samples)
    self.assertEqual(10, len(sampled_index_sequences))
    for output_sequence in sampled_index_sequences:
      self.assertEqual((5,), output_sequence.shape)
      self.assertEqual(4, len(set(output_sequence.tolist())))


class TestResizeSequence(unittest.TestCase):
  """Tests for utils.resize_sequence()"""

  def test_resize_sequence_without_permutation1(self):
    """Test when we do not permute, output is deterministic."""
    sub_sequence, seq_lengths, _, _ = utils.resize_sequence(
        sequence=np.array([[1, 1], [2, 2], [1, 1]]),
        cluster_id=np.array([1, 2, 1]),
        num_permutations=None)
    self.assertEqual(len(sub_sequence), 2)
    self.assertTrue((sub_sequence[0] == [[1, 1], [1, 1]]).all())
    self.assertTrue((sub_sequence[1] == [[2, 2]]).all())
    self.assertListEqual(seq_lengths, [3, 2])

  def test_resize_sequence_without_permutation2(self):
    """Test when we do not permute, output is deterministic."""
    sub_sequence, seq_lengths, _, _ = utils.resize_sequence(
        sequence=np.array([[1, 1], [2, 2], [3, 3]]),
        cluster_id=np.array([1, 2, 1]),
        num_permutations=None)
    self.assertEqual(len(sub_sequence), 2)
    self.assertTrue((sub_sequence[0] == [[1, 1], [3, 3]]).all())
    self.assertTrue((sub_sequence[1] == [[2, 2]]).all())
    self.assertListEqual(seq_lengths, [3, 2])

  def test_resize_sequence_with_permutation(self):
    """Test when we permute, each output can be one of the permutations."""
    sub_sequence, seq_lengths, _, _ = utils.resize_sequence(
        sequence=np.array([[1, 1], [2, 2], [3, 3]]),
        cluster_id=np.array([1, 2, 1]),
        num_permutations=2)
    self.assertEqual(len(sub_sequence), 2 * 2)
    self.assertTrue((sub_sequence[0] == [[1, 1], [3, 3]]).all() or
                    (sub_sequence[0] == [[3, 3], [1, 1]]).all())
    self.assertTrue((sub_sequence[1] == [[1, 1], [3, 3]]).all() or
                    (sub_sequence[1] == [[3, 3], [1, 1]]).all())
    self.assertTrue((sub_sequence[2] == [[2, 2]]).all())
    self.assertTrue((sub_sequence[3] == [[2, 2]]).all())
    self.assertListEqual(seq_lengths, [3, 3, 2, 2])


if __name__ == '__main__':
  unittest.main()
