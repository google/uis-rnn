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
"""Arguments for UISRNN."""
import argparse

_DEFAULT_OBSERVATION_DIM = 256


def parse_arguments():
  """Parse arguments."""
  parser = argparse.ArgumentParser(
      description='UIS-RNN model for speaker diarization.')

  # data configurations
  parser.add_argument(
      '--observation_dim',
      default=_DEFAULT_OBSERVATION_DIM,
      type=int,
      help='The dimension of the embeddings (e.g. d-vectors).')

  # model configurations
  parser.add_argument(
      '--rnn_hidden_size',
      default=512,
      type=int,
      help='The number of nodes for each RNN layer.')
  parser.add_argument(
      '--rnn_depth',
      default=1,
      type=int,
      help='The number of RNN layers.')
  parser.add_argument(
      '--rnn_dropout',
      default=0.2,
      type=float,
      help='The dropout rate for all RNN layers.')
  parser.add_argument(
      '--regularization_weight',
      '-r',
      default=1e-5,
      type=float,
      help='The network regularization multiplicative.')
  parser.add_argument(
      '--transition_bias',
      default=None,
      type=float,
      help='The value of p0, corresponding to Eq. (6) in the '
           'paper. If the value is given, we will fix to this value. If the '
           'value is None, we will estimate it from training data '
           'using Eq. (13) in the paper.')
  parser.add_argument(
      '--crp_alpha',
      default=1.0,
      type=float,
      help='The value of alpha for the Chinese restaurant process (CRP), '
           'corresponding to Eq. (7) in the paper. In this open source '
           'implementation, currently we only support using a given value '
           'of crp_alpha.')
  parser.add_argument(
      '--sigma2',
      default=None,
      type=float,
      help='The value of sigma squared, corresponding to Eq. (11) in the '
           'paper. If the value is given, we will fix to this value. If the '
           'value is None, we will estimate it from training data.')
  parser.add_argument(
      '--sigma_alpha',
      default=1.0,
      type=float,
      help='The inverse gamma shape for estimating sigma2. This value is only '
           'meaningful when sigma2 is not given, and estimated from data.')
  parser.add_argument(
      '--sigma_beta',
      default=1.0,
      type=float,
      help='The inverse gamma scale for estimating sigma2. This value is only '
           'meaningful when sigma2 is not given, and estimated from data.')

  # training configurations
  parser.add_argument(
      '--optimizer',
      '-o',
      default='adam',
      choices=['adam'],
      help='The optimizer for training.')
  parser.add_argument(
      '--learning_rate',
      '-l',
      default=1e-5,
      type=float,
      help='The leaning rate for training.')
  parser.add_argument(
      '--train_iteration',
      '-t',
      default=20000,
      type=int,
      help='The total number of training iterations.')
  parser.add_argument(
      '--batch_size',
      '-b',
      default=10,
      type=int,
      help='The batch size for training.')
  parser.add_argument(
      '--num_permutations',
      default=10,
      type=int,
      help='The number of permutations per utterance sampled in the training '
           'data.')

  # inference configurations
  parser.add_argument(
      '--beam_size',
      '-s',
      default=10,
      type=int,
      help='The beam search size for inference.')
  parser.add_argument(
      '--look_ahead',
      default=1,
      type=int,
      help='The number of look ahead steps during inference.')
  parser.add_argument(
      '--test_iteration',
      default=2,
      type=int,
      help='During inference, we concatenate M duplicates of the test '
           'sequence, and run inference on this concatenated sequence. '
           'Then we return the inference results on the last duplicate as the '
           'final prediction for the test sequence.')

  args = parser.parse_args()
  return args
