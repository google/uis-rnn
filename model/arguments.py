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

_TOY_DATA_D_OBSERVATION = 256


def parse_arguments():
  """Parse arguments."""
  parser = argparse.ArgumentParser(
      description='Bayesian Non-parametric Model For Diarization')
  # data configurations
  parser.add_argument(
      '--dataset', '-d', default='toy', type=str, help='dataset type')
  parser.add_argument(
      '--d_observation',
      default=_TOY_DATA_D_OBSERVATION,
      type=int,
      help='data dimension')
  # model configurations
  parser.add_argument(
      '--rnn_hidden_size',
      default=256,
      type=int,
      help='rnn hidden state dimension')
  parser.add_argument('--rnn_depth', default=1, type=int, help='rnn depth')
  parser.add_argument(
      '--rnn_dropout', default=0.2, type=float, help='rnn dropout rate')
  parser.add_argument(
      '--regularization_weight',
      '-r',
      default=1e-5,
      type=float,
      help='network regularization multiplicative')
  parser.add_argument(
      '--alpha', default=1.0, type=float, help='inverse gamma shape')
  parser.add_argument(
      '--beta', default=1.0, type=float, help='inverse gamma scale')
  parser.add_argument(
      '--crp_theta', default=1.0, type=float, help='crp parameter')
  parser.add_argument(
      '--sigma2',
      default=.05,
      type=float,
      help='update sigma2 if it equals to None')
  # training/testing configurations
  parser.add_argument(
      '--optimizer', '-o', default='adam', choices=['adam'], help='optimizer')
  parser.add_argument(
      '--learning_rate', '-l', default=1e-5, type=float, help='leaning rate')
  parser.add_argument(
      '--train_iteration',
      '-t',
      default=20000,
      type=int,
      help='total training iteration')
  parser.add_argument(
      '--test_iteration', default=2, type=int, help='total testing iteration')
  parser.add_argument(
      '--batch_size', '-b', default=10, type=int, help='batch size')
  parser.add_argument(
      '--beam_size', '-s', default=10, type=int, help='beam search size')
  parser.add_argument(
      '--look_ahead', default=1, type=int, help='look ahead steps in testing')
  parser.add_argument(
      '--num_permutations',
      default=10,
      type=int,
      help='number of permutations per utterance sampled in the training data')
  parser.add_argument(
      '--pretrain', default=None, type=str, help='use pretrained model')

  args = parser.parse_args()
  return args
