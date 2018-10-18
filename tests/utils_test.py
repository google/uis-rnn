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

from model.utils import sample_permuted_segments


class TestSamplePermutedSegments(unittest.TestCase):

  def test_short_sequence(self):
    index_sequence = [5, 2, 3, 2, 1]
    number_samples = 10
    sampled_index_sequences = sample_permuted_segments(index_sequence,
                                                       number_samples)
    self.assertEqual(10, len(sampled_index_sequences))
    for output_sequence in sampled_index_sequences:
      self.assertEqual((5,), output_sequence.shape)
      self.assertEqual(4, len(set(output_sequence.tolist())))


if __name__ == '__main__':
  unittest.main()
