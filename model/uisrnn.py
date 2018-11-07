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
"""The UISRNN model."""
from model import utils
import numpy as np
import os
import tempfile
import torch
from torch import autograd
from torch import nn
from torch import optim
import torch.nn.functional as F
import zipfile

_INITIAL_SIGMA2_VALUE = 0.1
_SAVED_STATES_FILE = 'saved_model.states'
_SAVED_NPZ_FILE = 'saved_model.npz'


class NormalRNN(nn.Module):
  """Normal Recurent Neural Networks."""

  def __init__(self, input_dim, hidden_size, depth, dropout, observation_dim):
    super(NormalRNN, self).__init__()
    self.hidden_size = hidden_size
    if depth >= 2:
      self.gru = nn.GRU(input_dim, hidden_size, depth, dropout=dropout)
    else:
      self.gru = nn.GRU(input_dim, hidden_size, depth)
    self.linear_mean1 = nn.Linear(hidden_size, hidden_size)
    self.linear_mean2 = nn.Linear(hidden_size, observation_dim)

  def forward(self, input_seq, hidden=None):
    output_seq, hidden = self.gru(input_seq, hidden)
    if isinstance(output_seq, torch.nn.utils.rnn.PackedSequence):
      output_seq, _ = torch.nn.utils.rnn.pad_packed_sequence(
          output_seq, batch_first=False)
    mean = self.linear_mean2(F.relu(self.linear_mean1(output_seq)))
    return mean, hidden


class UISRNN(object):
  """Unbounded Interleaved-State Recurrent Neural Networks """

  def __init__(self, args):
    """Construct the UISRNN object.

    Args:
      args: return value of arguments.parse_arguments()
    """
    self.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')
    self.rnn_model = NormalRNN(args.observation_dim, args.rnn_hidden_size,
                               args.rnn_depth, args.rnn_dropout,
                               args.observation_dim).to(self.device)
    self.rnn_init_hidden = nn.Parameter(
        torch.zeros(args.rnn_depth, 1, args.rnn_hidden_size).to(self.device))
    sigma2 = _INITIAL_SIGMA2_VALUE if args.sigma2 is None else args.sigma2
    self.sigma2 = nn.Parameter(
        sigma2 * torch.ones(args.observation_dim).to(self.device))
    self.transition_bias = args.transition_bias

  def _get_optimizer(self, optimizer, sigma2, learning_rate):
    """Get optimizer for UISRNN.

    Args:
      optimizer: string - name of the optimizer.
      sigma2: - variance parameters.
        We can either train sigma2 by setting "sigma2=None",
        or we can fix it by setting sigma2 equals to a scalar in args.
      learning_rate: - learning rate for the entire model.
        We do not customize learning rate for separate parts.

    Returns:
      a pytorch "optim" object
    """
    params = [
        {
            'params': self.rnn_model.parameters()
        },  # rnn parameters
        {
            'params': self.rnn_init_hidden
        }  # rnn initial hidden state
    ]
    if sigma2 is None:  # train sigma2
      params.append({
          'params': self.sigma2
      }  # variance parameters
                   )
    assert optimizer == 'adam', 'Only adam optimizer is supported.'
    return optim.Adam(params, lr=learning_rate)

  def save(self, filepath):
    """Save the model to a file.

    Args:
      filepath: the path of the file.
    """
    tempdir = tempfile.mkdtemp()
    # save states
    states_file = os.path.join(tempdir, _SAVED_STATES_FILE)
    torch.save(self.rnn_model.state_dict(), states_file)

    # save other parameters
    npz_file = os.path.join(tempdir, _SAVED_NPZ_FILE)
    np.savez(npz_file,
             transition_bias=self.transition_bias,
             sigma2=self.sigma2.detach().cpu().numpy())

    # create combined model file
    with zipfile.ZipFile(filepath, 'w') as myzip:
      myzip.write(states_file, _SAVED_STATES_FILE)
      myzip.write(npz_file, _SAVED_NPZ_FILE)

  def load(self, filepath):
    """Load the model from a file.

    Args:
      filepath: the path of the file.
    """
    tempdir = tempfile.mkdtemp()
    # extract zip file
    with zipfile.ZipFile(filepath) as myzip:
      myzip.extract(_SAVED_STATES_FILE, path=tempdir)
      myzip.extract(_SAVED_NPZ_FILE, path=tempdir)

    # load states
    states_file = os.path.join(tempdir, _SAVED_STATES_FILE)
    self.rnn_model.load_state_dict(torch.load(states_file))

    # load other parameters
    npz_file = os.path.join(tempdir, _SAVED_NPZ_FILE)
    data = np.load(npz_file)
    self.transition_bias = float(data['transition_bias'])
    self.sigma2 = nn.Parameter(
        torch.from_numpy(data['sigma2']).to(self.device))

  def fit(self, args, train_sequence, train_cluster_id):
    """Fit UISRNN model.

    Args:
      args: Model and training configurations. See demo for description.
      train_sequence: (real 2d numpy array, size: N by D)
        - the training d_vector sequence.
        N - summation of lengths of all utterances
        D - observation dimension
        For example, train_sequence =
        [[1.2 3.0 -4.1 6.0]    --> an entry of speaker #0 from utterance 'iaaa'
         [0.8 -1.1 0.4 0.5]    --> an entry of speaker #1 from utterance 'iaaa'
         [-0.2 1.0 3.8 5.7]    --> an entry of speaker #0 from utterance 'iaaa'
         [3.8 -0.1 1.5 2.3]    --> an entry of speaker #0 from utterance 'ibbb'
         [1.2 1.4 3.6 -2.7]]   --> an entry of speaker #0 from utterance 'ibbb'
        Here N=5, d=4.
        We concatenate all training utterances into a single sequence.
      train_cluster_id: (a vector of strings, size: N)
        - the speaker id sequence.
        For example, train_cluster_id =
        ['iaaa_0', 'iaaa_1', 'iaaa_0', 'ibbb_0', 'ibbb_0']
        'iaaa_0' means the entry belongs to speaker #0 in utterance 'iaaa'.
        Note that the order of entries within an utterance are preserved,
        and all utterances are simply concatenated together.

    Raises:
      ValueError: If train_sequence has wrong dimension.
    """

    train_total_length, observation_dim = train_sequence.shape
    if observation_dim != args.observation_dim:
      raise ValueError('train_sequence does not match the dimension specified '
                       'by args.observation_dim.')
    if train_total_length != len(train_cluster_id):
      raise ValueError('train_sequence length is not equal to '
                       'train_cluster_id length.')
    if type(train_sequence).__module__ != np.__name__:
      raise TypeError('train_sequence type should be an numpy array.')
    if type(train_cluster_id).__module__ != np.__name__:
      raise TypeError('train_cluster_id type should be an numpy array.')

    self.rnn_model.train()
    optimizer = self._get_optimizer(optimizer=args.optimizer,
                                    sigma2=args.sigma2,
                                    learning_rate=args.learning_rate)

    sub_sequences, seq_lengths, transition_bias = utils.resize_sequence(
        sequence=train_sequence,
        cluster_id=train_cluster_id,
        num_permutations=args.num_permutations)
    num_clusters = len(seq_lengths)
    sorted_seq_lengths = np.sort(seq_lengths)[::-1]
    permute_index = np.argsort(seq_lengths)[::-1]
    if self.transition_bias is None:
      self.transition_bias = transition_bias
    if args.batch_size is None:
      # Packing sequences.
      rnn_input = np.zeros((sorted_seq_lengths[0], num_clusters,
                            args.observation_dim))
      for i in range(num_clusters):
        rnn_input[1:sorted_seq_lengths[i], i, :] = sub_sequences[
            permute_index[i]]
      rnn_input = autograd.Variable(
        torch.from_numpy(rnn_input).float()).to(self.device)
      packed_train_sequence, rnn_truth = utils.pack_seq(rnn_input,
                                                        sorted_seq_lengths)

    train_loss = []
    for t in range(args.train_iteration):
      optimizer.zero_grad()
      if args.batch_size is not None:
        mini_batch = np.sort(np.random.choice(num_clusters, args.batch_size))
        mini_batch_rnn_input = np.zeros((sorted_seq_lengths[mini_batch[0]],
                                         args.batch_size, args.observation_dim))
        for i in range(args.batch_size):
          mini_batch_rnn_input[1:sorted_seq_lengths[mini_batch[i]],
                               i, :] = sub_sequences[permute_index[
                                   mini_batch[i]]]
        mini_batch_rnn_input = autograd.Variable(
            torch.from_numpy(mini_batch_rnn_input).float()).to(self.device)
        packed_train_sequence, rnn_truth = utils.pack_seq(
            mini_batch_rnn_input, sorted_seq_lengths[mini_batch])

      hidden = self.rnn_init_hidden.repeat(1, args.batch_size, 1)
      mean, _ = self.rnn_model(packed_train_sequence, hidden)
      # use mean to predict
      mean = torch.cumsum(mean, dim=0)
      mean_size = mean.size()
      mean = torch.mm(
          torch.diag(
            1.0 / torch.arange(1, mean_size[0] + 1).float().to(self.device)),
          mean.view(mean_size[0], -1))
      mean = mean.view(mean_size)

      # Likelihood part.
      loss1 = utils.weighted_mse_loss(
          input_tensor=(rnn_truth != 0).float() * mean[:-1, :, :],
          target_tensor=rnn_truth,
          weight=1 / (2 * self.sigma2))

      weight = (((rnn_truth != 0).float() * mean[:-1, :, :] - rnn_truth)
                **2).view(-1, observation_dim)
      num_non_zero = torch.sum((weight != 0).float(), dim=0).squeeze()
      loss2 = ((2 * args.sigma_alpha + num_non_zero + 2) /
               (2 * num_non_zero) * torch.log(self.sigma2)).sum() + (
                   args.sigma_beta / (self.sigma2 * num_non_zero)).sum()
      # regularization
      l2_reg = 0
      for param in self.rnn_model.parameters():
        l2_reg += torch.norm(param)
      loss3 = args.regularization_weight * l2_reg

      loss = loss1 + loss2 + loss3
      loss.backward()
      nn.utils.clip_grad_norm_(self.rnn_model.parameters(), 5.0)
      # nn.utils.clip_grad_norm_(self.sigma2, 1.0)
      optimizer.step()
      # avoid numerical issues
      self.sigma2.data.clamp_(min=1e-6)

      if np.remainder(t, 10) == 0:
        print('Iter: {:d}  \t'
              'Training Loss: {:.4f}    \n'
              '    Negative Log Likelihood: {:.4f}\t'
              'Sigma2 Prior: {:.4f}\t'
              'Regularization: {:.4f}'.format(t, float(loss.data),
                float(loss1.data), float(loss2.data), float(loss3.data)))
      train_loss.append(float(loss1.data))  # only save the likelihood part
    print('Done training with {} iterations'.format(args.train_iteration))

  def predict(self, args, test_sequence):
    """Predict test sequence labels using UISRNN model.

    Args:
      args: Model and testing configurations. See demo for description.
      test_sequence: (real 2d numpy array, size: N by D)
        - the test d_vector sequence.
        N - length of one test utterance
        D - observation dimension
        For example, test_sequence =
        [[2.2 -1.0 3.0 5.6]    --> 1st entry of utterance 'iccc'
         [0.5 1.8 -3.2 0.4]    --> 2nd entry of utterance 'iccc'
         [-2.2 5.0 1.8 3.7]    --> 3rd entry of utterance 'iccc'
         [-3.8 0.1 1.4 3.3]    --> 4th entry of utterance 'iccc'
         [0.1 2.7 3.5 -1.7]]   --> 5th entry of utterance 'iccc'
        Here N=5, d=4.

    Returns:
      predicted_cluster_id: (integer array, size: N)
        - predicted speaker id sequence.
        For example, predicted_cluster_id = [0, 1, 0, 0, 1]

    Raises:
      ValueError: If test_sequence has wrong dimension.
    """
    test_sequence_length, observation_dim = test_sequence.shape
    if observation_dim != args.observation_dim:
      raise ValueError('test_sequence does not match the dimension specified '
                       'by args.observation_dim.')
    if type(test_sequence).__module__ != np.__name__:
      raise TypeError('test_sequence type should be an numpy array.')
    self.rnn_model.eval()
    test_sequence = np.tile(test_sequence, (args.test_iteration, 1))
    test_sequence = autograd.Variable(
        torch.from_numpy(test_sequence).float()).to(self.device)
    # bookkeeping for beam search
    # each cell consists of:
    # (mean_set, hidden_set, score/-likelihood, trace, block_counts)
    proposal_set = [([], [], 0, [], [])]
    max_clusters = 0

    for t in np.arange(0, args.test_iteration * test_sequence_length,
                       args.look_ahead):
      l_remain = args.test_iteration * test_sequence_length - t
      score_set = float('inf') * np.ones(
          np.append(
              args.beam_size, max_clusters + 1 + np.arange(
                  np.min([l_remain, args.look_ahead]))))
      for proposal_rank, proposal in enumerate(proposal_set):
        mean_buffer = list(proposal[0])
        hidden_buffer = list(proposal[1])
        score_buffer = proposal[2]
        trace_buffer = proposal[3]
        block_counts_buffer = list(proposal[4])
        n_clusters = len(mean_buffer)
        proposal_score_subset = float('inf') * np.ones(
            n_clusters + 1 + np.arange(np.min([l_remain, args.look_ahead])))
        for cluster_seq, _ in np.ndenumerate(proposal_score_subset):
          new_mean_buffer = mean_buffer.copy()
          new_hidden_buffer = hidden_buffer.copy()
          new_trace_buffer = trace_buffer.copy()
          new_block_counts_buffer = block_counts_buffer.copy()
          new_n_clusters = n_clusters
          new_loss = 0
          update_score = True
          for sub_idx, cluster in enumerate(cluster_seq):
            if cluster > new_n_clusters:  # invalid trace
              update_score = False
              break
            if cluster < new_n_clusters:  # existing clusters
              new_last_cluster = new_trace_buffer[-1]
              loss = utils.weighted_mse_loss(
                  input_tensor=torch.squeeze(new_mean_buffer[cluster]),
                  target_tensor=test_sequence[t + sub_idx, :],
                  weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
              if cluster == new_last_cluster:
                loss -= np.log(1 - self.transition_bias)
              else:
                loss -= np.log(self.transition_bias) + np.log(
                    new_block_counts_buffer[cluster]) - np.log(
                        sum(new_block_counts_buffer) + args.crp_alpha)
              # update new mean and new hidden
              mean, hidden = self.rnn_model(
                  test_sequence[t + sub_idx, :].unsqueeze(0).unsqueeze(0),
                  new_hidden_buffer[cluster])
              new_mean_buffer[cluster] = (new_mean_buffer[cluster] * (
                  (np.array(new_trace_buffer) == cluster).sum() -
                  1).astype(float) + mean.clone()) / (
                      np.array(new_trace_buffer) == cluster).sum().astype(
                          float)  # use mean to predict
              new_hidden_buffer[cluster] = hidden.clone()
              if cluster != new_trace_buffer[-1]:
                new_block_counts_buffer[cluster] += 1
              new_trace_buffer.append(cluster)
            else:  # new cluster
              init_input = autograd.Variable(
                  torch.zeros(args.observation_dim)
                  ).unsqueeze(0).unsqueeze(0).to(self.device)
              mean, hidden = self.rnn_model(init_input,
                                            self.rnn_init_hidden)
              loss = utils.weighted_mse_loss(
                  input_tensor=torch.squeeze(mean),
                  target_tensor=test_sequence[t + sub_idx, :],
                  weight=1 / (2 * self.sigma2)).cpu().detach().numpy()
              loss -= np.log(self.transition_bias) + np.log(
                  args.crp_alpha) - np.log(
                      sum(new_block_counts_buffer) + args.crp_alpha)
              # update new min and new hidden
              mean, hidden = self.rnn_model(
                  test_sequence[t + sub_idx, :].unsqueeze(0).unsqueeze(0),
                  hidden)
              new_mean_buffer.append(mean.clone())
              new_hidden_buffer.append(hidden.clone())
              new_block_counts_buffer.append(1)
              new_trace_buffer.append(cluster)
              new_n_clusters += 1
            new_loss += loss
          if update_score:
            score_set[tuple([proposal_rank]) +
                      cluster_seq] = score_buffer + new_loss

      # find top scores
      score_ranked = np.sort(score_set, axis=None)
      score_ranked[score_ranked == float('inf')] = 0
      score_ranked = np.trim_zeros(score_ranked)
      idx_ranked = np.argsort(score_set, axis=None)

      # update best traces
      new_proposal_set = []
      max_clusters = 0
      for new_proposal_rank in range(
          np.min((len(score_ranked), args.beam_size))):
        total_idx = np.unravel_index(idx_ranked[new_proposal_rank],
                                     score_set.shape)
        prev_proposal_idx = total_idx[0]
        new_cluster_idx = total_idx[1:]
        (mean_set, hidden_set, _, trace,
         block_counts) = proposal_set[prev_proposal_idx]
        new_mean_set = mean_set.copy()
        new_hidden_set = hidden_set.copy()
        new_score = score_ranked[
            new_proposal_rank]  # can safely update the likelihood for now
        new_trace = trace.copy()
        new_block_counts = block_counts.copy()
        new_n_clusters = len(new_mean_set)
        max_clusters = max(max_clusters, new_n_clusters)
        for sub_idx, cluster in enumerate(
            new_cluster_idx):  # update the proposal step-by-step
          if cluster == new_n_clusters:
            init_input = autograd.Variable(
                torch.zeros(args.observation_dim)
                ).unsqueeze(0).unsqueeze(0).to(self.device)
            mean, hidden = self.rnn_model(init_input,
                                          self.rnn_init_hidden)
            mean, hidden = self.rnn_model(
                test_sequence[t + sub_idx, :].unsqueeze(0).unsqueeze(0), hidden)
            new_mean_set.append(mean.clone())
            new_hidden_set.append(hidden.clone())
            new_block_counts.append(1)
            new_trace.append(cluster)
            new_n_clusters += 1
            max_clusters = max(max_clusters, new_n_clusters)
          else:
            mean, hidden = self.rnn_model(
                test_sequence[t + sub_idx, :].unsqueeze(0).unsqueeze(0),
                new_hidden_set[cluster])
            new_mean_set[cluster] = (
                new_mean_set[cluster] * (
                    (np.array(new_trace) == cluster).sum() - 1).astype(float) +
                mean.clone()) / (np.array(new_trace) == cluster).sum().astype(
                    float)  # use mean to predict
            new_hidden_set[cluster] = hidden.clone()
            if cluster != new_trace[-1]:
              new_block_counts[cluster] += 1
            new_trace.append(cluster)
        new_proposal_set.append((new_mean_set, new_hidden_set, new_score,
                                 new_trace, new_block_counts))
      proposal_set = new_proposal_set

    predicted_cluster_id = proposal_set[0][3][-test_sequence_length:]
    return predicted_cluster_id
