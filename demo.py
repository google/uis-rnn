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

from model.arguments import parse_arguments
from model.evals import evaluate_result
from model.uisrnn import UISRNN
from model.utils import output_result
import numpy as np


def diarization_experiment(args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    args: return value of parser.parse_args()
  """

  predict_labels = []
  test_record = []

  train_data = np.load('./data/training_data.npz')
  test_data = np.load('./data/testing_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  test_sequences = test_data['test_sequences']
  test_cluster_ids = test_data['test_cluster_ids']

  _, observation_dim = train_sequence.shape
  input_dim = observation_dim

  model = UISRNN(args, input_dim, observation_dim, .5)
  # training
  if args.pretrain is None:
    model.fit(args, train_sequence, train_cluster_id)
    model.save(args)
  else:  # use pretrained model
    model.load(args)

  # testing
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    predict_label = model.predict(args, test_sequence)
    predict_labels.append(predict_label)
    accuracy, length = evaluate_result(test_cluster_id, predict_label)
    test_record.append((accuracy, length))
    print('ground truth labels:')
    print(test_cluster_id)
    print('predict labels:')
    print(predict_label)
    print('----------------------')

  output_result(args, test_record)

  print('Finish --dataset {} --alpha {} --beta {} --crp_theta {} -l {} -r {}'
        .format(args.dataset, args.alpha, args.beta, args.crp_theta,
                args.learn_rate, args.network_reg))


def main():
  # fix random seeds for reproducing results
  # np.random.seed(1)
  # torch.manual_seed(1)
  # torch.cuda.manual_seed(1)

  args = parse_arguments()

  diarization_experiment(args)


if __name__ == '__main__':
  main()
