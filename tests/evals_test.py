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

import unittest

from model import evals


class TestComputeApproximateSequenceAcuracy(unittest.TestCase):

  def test_mismatched_sequences(self):
    sequence1 = [0, 0, 1, 2, 2]
    sequence2 = [3, 3, 4, 4, 1]
    accuracy = evals.compute_approximate_sequence_accuracy(sequence1,
                                                           sequence2)
    self.assertEqual(0.8, accuracy)

  def test_equivalent_sequences(self):
    sequence1 = [0, 0, 1, 2, 2]
    sequence2 = [3, 3, 4, 1, 1]
    accuracy = evals.compute_approximate_sequence_accuracy(sequence1,
                                                           sequence2)
    self.assertEqual(1.0, accuracy)

  def test_sequences_of_different_lengths(self):
    sequence1 = [0, 0, 1, 2]
    sequence2 = [3, 3, 4, 4, 1]
    with self.assertRaises(Exception):
      evals.compute_approximate_sequence_accuracy(sequence1, sequence2)

  def test_empty_sequences(self):
    with self.assertRaises(Exception):
      evals.compute_approximate_sequence_accuracy([], [])


class TestEvaluateResult(unittest.TestCase):

  def test_mismatched_sequences(self):
    sequence1 = [0, 0, 1, 2, 2]
    sequence2 = [3, 3, 4, 4, 1]
    accuracy, length = evals.evaluate_result(sequence1, sequence2)
    self.assertEqual(0.8, accuracy)
    self.assertEqual(5, length)

  def test_equivalent_sequences(self):
    sequence1 = [0, 0, 1, 2, 2]
    sequence2 = [3, 3, 4, 1, 1]
    accuracy, length = evals.evaluate_result(sequence1, sequence2)
    self.assertEqual(1.0, accuracy)
    self.assertEqual(5, length)


if __name__ == '__main__':
  unittest.main()
