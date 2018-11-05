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

from model import arguments
from model import evals
from model import uisrnn
from model import utils
import numpy as np

SAVED_STATES_FILE_NAME = 'saved_uisrnn_states.model'


def diarization_experiment(args):
  """Experiment pipeline.

  Load data --> train model --> test model --> output result

  Args:
    args: return value of arguments.parse_arguments()
  """

  predicted_labels = []
  test_record = []

  train_data = np.load('./data/training_data.npz')
  test_data = np.load('./data/testing_data.npz')
  train_sequence = train_data['train_sequence']
  train_cluster_id = train_data['train_cluster_id']
  test_sequences = test_data['test_sequences']
  test_cluster_ids = test_data['test_cluster_ids']

  model = uisrnn.UISRNN(args, .5)
  # training
  if args.pretrain is None:
    model.fit(args, train_sequence, train_cluster_id)
    model.save(SAVED_STATES_FILE_NAME)
  else:  # use pretrained model
    # TODO: support using pretrained model.
    model.load(SAVED_STATES_FILE_NAME)

  # testing
  for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
    predicted_label = model.predict(args, test_sequence)
    predicted_labels.append(predicted_label)
    accuracy, length = evals.evaluate_result(test_cluster_id, predicted_label)
    test_record.append((accuracy, length))
    print('Ground truth labels:')
    print(test_cluster_id)
    print('Predicted labels:')
    print(predicted_label)
    print('----------------------')

  utils.output_result(args, test_record)

  print('Finish --alpha {} --beta {} --crp_theta {} -l {} -r {}'
        .format(args.alpha, args.beta, args.crp_theta,
                args.learning_rate, args.regularization_weight))


def main():
  # fix random seeds for reproducing results
  # np.random.seed(1)
  # torch.manual_seed(1)
  # torch.cuda.manual_seed(1)

  args = arguments.parse_arguments()

  diarization_experiment(args)


if __name__ == '__main__':
  main()
