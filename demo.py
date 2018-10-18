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

import argparse
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model.uisrnn import UISRNN
from model.utils import output_result
from model.eval import evaluate_result


def diarization_experiment(args):
  '''Experiment pipeline

  Load data --> train model --> test model --> output result
  '''

  predict_labels = []
  test_record = []

  train_data = np.load('./data/training_data.npz')
  test_data = np.load('./data/testing_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  test_sequences = test_data['test_sequences']
  test_cluster_ids = test_data['test_cluster_ids']

  _ , observation_dim = train_sequence.shape
  input_dim = observation_dim

  model = UISRNN(args, input_dim, observation_dim, .5)
  # training
  if args.pretrain == None:
    model.fit(args, train_sequence, train_cluster_id)
    model.save(args)
  else: # use pretrained model
    model.load(args)

  # testing
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    predict_label = model.predict(args, test_sequence)
    predict_labels.append(predict_label)
    accuracy, length = evaluate_result(args, test_cluster_id, predict_label)
    test_record.append((accuracy, length))
    print('ground truth labels:')
    print(test_cluster_id)
    print('predict labels:')
    print(predict_label)
    print('----------------------')

  output_result(args, test_record)

  print('Finish --dataset {} --alpha {} --beta {} --crp_theta {} -l {} -r {}'.format(
    args.dataset, args.alpha, args.beta, args.crp_theta, args.learn_rate, args.network_reg))


if __name__ == '__main__':

  # fix random seeds for reproducing results
  # np.random.seed(1)
  # torch.manual_seed(1)
  # torch.cuda.manual_seed(1)

  parser = argparse.ArgumentParser(description='Bayesian Non-parametric Model For Diarization')
  # data configurations
  parser.add_argument('--dataset', '-d', default='toy', type=str, help='dataset type')
  parser.add_argument('--toy_data_d_observation', default=256, type=int, help='toy data dimension')
  # model configurations
  parser.add_argument('--model_type', '-m', default='generative', type=str, help='model type')
  parser.add_argument('--rnn_hidden_size', default=256, type=int, help='rnn hidden state dimension')
  parser.add_argument('--rnn_depth', default=1, type=int, help='rnn depth')
  parser.add_argument('--rnn_dropout', default=0.2, type=float, help='rnn dropout rate')
  parser.add_argument('--network_reg', '-r', default=1e-5, type=float, help='network regularization multiplicative')
  parser.add_argument('--alpha', default=1.0, type=float, help='inverse gamma shape')
  parser.add_argument('--beta', default=1.0, type=float, help='inverse gamma scale')
  parser.add_argument('--crp_theta', default=1.0, type=float, help='crp parameter')
  parser.add_argument('--sigma2', default=.05, type=float, help='update sigma2 if it equals to None')
  # training/testing configurations
  parser.add_argument('--optimizer', '-o', default='adam', type=str, help='optimizer')
  parser.add_argument('--learn_rate', '-l', default=1e-5, type=float, help='leaning rate')
  parser.add_argument('--train_iteration', '-t', default=20000, type=int, help='total training iteration')
  parser.add_argument('--test_iteration', default=2, type=int, help='total testing iteration')
  parser.add_argument('--batch_size', '-b', default=10, type=int, help='batch size')
  parser.add_argument('--beam_size', '-s', default=10, type=int, help='beam search size')
  parser.add_argument('--look_ahead', default=1, type=int, help='look ahead steps in testing')
  parser.add_argument('--permutation', default=10, type=int, help='number of permutations per utterance sampled in the training data')
  parser.add_argument('--pretrain', default=None, type=str, help='use pretrained model')

  args = parser.parse_args()

  diarization_experiment(args)
