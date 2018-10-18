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

import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from model.utils import pack_seq, resize_seq, weighted_mse_loss


class NormalRNN(nn.Module):
  def __init__(self, input_dim, hidden_size, depth, dropout, observation_dim):
    super(NormalRNN, self).__init__()
    self.hidden_size = hidden_size
    if depth >= 2:
      self.gru = nn.GRU(input_dim, hidden_size, depth, dropout=dropout)
    else:
      self.gru = nn.GRU(input_dim, hidden_size, depth)
    self.linear_mean1 = nn.Linear(hidden_size, hidden_size)
    self.linear_mean2 = nn.Linear(hidden_size, observation_dim)

  def forward(self, input, hidden=None):
    output, hidden = self.gru(input, hidden)
    if isinstance(output, torch.nn.utils.rnn.PackedSequence):
      output, output_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
    mean = self.linear_mean2(F.relu(self.linear_mean1(output)))
    return mean, hidden


class UISRNN():
  def __init__(self, args, input_dim, observation_dim, transition_bias):
    if torch.cuda.is_available():
      self.rnn_model = NormalRNN(input_dim, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout, observation_dim).cuda()
      self.rnn_init_hidden = nn.Parameter(torch.zeros(1, args.rnn_hidden_size).cuda())
      if args.sigma2 == None:
        self.sigma2 = nn.Parameter(.1*torch.ones(observation_dim).cuda())
      else:
        self.sigma2 = nn.Parameter(args.sigma2*torch.ones(observation_dim).cuda())
    else:
      self.rnn_model = NormalRNN(input_dim, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout, observation_dim)
      self.rnn_init_hidden = nn.Parameter(torch.zeros(1, args.rnn_hidden_size))
      if args.sigma2 == None:
        self.sigma2 = nn.Parameter(.1*torch.ones(observation_dim))
      else:
        self.sigma2 = nn.Parameter(args.sigma2*torch.ones(observation_dim))
    self.transition_bias = transition_bias

  def save(self, args):
    torch.save(self.rnn_model.state_dict(), 'rnn_model {}'.format(args.dataset))

  def load(self, args):
    self.rnn_model.load_state_dict(torch.load('rnn_model {}'.format(args.dataset)))
  
  def fit(self, args, train_sequence, train_cluster_id):
    '''Fit UISRNN model.

    Args:
      args: Model and training configurations. See demo for description.

      train_sequence (real 2d numpy array, size: N by D): The training d_vector sequence.
        N: summation of lengths of all utterances
        D: observation dimension
        Example: 
          train_sequence= [[1.2 3.0 -4.1 6.0]    --> an entry of speaker #0 from utterance 'iaaa'
                           [0.8 -1.1 0.4 0.5]    --> an entry of speaker #1 from utterance 'iaaa'
                           [-0.2 1.0 3.8 5.7]    --> an entry of speaker #0 from utterance 'iaaa'
                           [3.8 -0.1 1.5 2.3]    --> an entry of speaker #0 from utterance 'ibbb'
                           [1.2 1.4 3.6 -2.7]]   --> an entry of speaker #0 from utterance 'ibbb'
          Here N=5, d=4.
        Note that we concatenate all training utterances into a single sequence.

      train_cluster_id (a vector of strings, size: N): The speaker id sequence.
        Example:
          train_speaker_id = ['iaaa_0', 'iaaa_1', 'iaaa_0', 'ibbb_0', 'ibbb_0']
          Here 'iaaa_0' means the entry belongs to 'speaker #0' in utterance 'iaaa'.
          Note that the order of entries within an utterance are preserved, and all utterances
          are simply concatenated together.
    '''

    if args.model_type == 'generative':
      _ , observation_dim = train_sequence.shape
      input_dim = observation_dim

      self.rnn_model.train()
      if args.optimizer == 'adam':
        if args.sigma2 == None: # train sigma2
          optimizer = optim.Adam([
            {'params': self.rnn_model.parameters()}, # rnn parameters
            {'params': self.rnn_init_hidden}, # rnn initial hidden state
            {'params': self.sigma2} # variance parameters
            ], lr=args.learn_rate)
        else: # don't train sigma2
          optimizer = optim.Adam([
            {'params': self.rnn_model.parameters()}, # rnn parameters
            {'params': self.rnn_init_hidden} # rnn initial hidden state
            ], lr=args.learn_rate)

      sub_sequences, seq_lengths, transition_bias = resize_seq(args, train_sequence, train_cluster_id)
      num_clusters = len(seq_lengths)
      sorted_seq_lengths = np.sort(seq_lengths)[::-1]
      permute_index = np.argsort(seq_lengths)[::-1]
      self.transition_bias = transition_bias
      if args.batch_size is None:
        rnn_input = np.zeros((sorted_seq_lengths[0],num_clusters,input_dim))
        for i in range(num_clusters):
          rnn_input[1:sorted_seq_lengths[i],i,:] = sub_sequences[permute_index[i]]
        rnn_input = Variable(torch.from_numpy(rnn_input).float())
        if torch.cuda.is_available():
          rnn_input = rnn_input.cuda()
        packed_train_sequence, rnn_truth = pack_seq(rnn_input, sorted_seq_lengths)

      train_loss = []
      for t in range(args.train_iteration):
        optimizer.zero_grad()
        if args.batch_size is not None:
          mini_batch = np.sort(np.random.choice(num_clusters, args.batch_size))
          mini_batch_rnn_input = np.zeros((sorted_seq_lengths[mini_batch[0]],args.batch_size,input_dim))
          for i in range(args.batch_size):
            mini_batch_rnn_input[1:sorted_seq_lengths[mini_batch[i]],i,:] = sub_sequences[permute_index[mini_batch[i]]]
          mini_batch_rnn_input = Variable(torch.from_numpy(mini_batch_rnn_input).float())
          if torch.cuda.is_available():
            mini_batch_rnn_input = mini_batch_rnn_input.cuda()
          packed_train_sequence, rnn_truth = pack_seq(mini_batch_rnn_input, sorted_seq_lengths[mini_batch])
        if torch.cuda.is_available():
          hidden = torch.mm(torch.ones(args.batch_size,1).float().cuda(), self.rnn_init_hidden).unsqueeze(0)
        else:
          hidden = torch.mm(torch.ones(args.batch_size,1).float(), self.rnn_init_hidden).unsqueeze(0)
        mean, _ = self.rnn_model(packed_train_sequence, hidden)
        # use mean to predict
        mean = torch.cumsum(mean, dim=0)
        mean_size = mean.size()
        if torch.cuda.is_available():
          mean = torch.mm(torch.diag(1.0/torch.arange(1,mean_size[0]+1).float().cuda()), mean.view(mean_size[0],-1))
        else:
          mean = torch.mm(torch.diag(1.0/torch.arange(1,mean_size[0]+1).float()), mean.view(mean_size[0],-1))
        mean = mean.view(mean_size)

        loss1 = weighted_mse_loss((rnn_truth!=0).float()*mean[:-1,:,:], rnn_truth, 1/(2*self.sigma2))
        weight = (((rnn_truth!=0).float()*mean[:-1,:,:] - rnn_truth)**2).view(-1,observation_dim)
        sum_weight = torch.sum(weight, dim=0).squeeze()
        num_non_zero = torch.sum((weight!=0).float(), dim=0).squeeze()
        loss2 = ((2*args.alpha+num_non_zero+2)/(2*num_non_zero)*torch.log(self.sigma2)).sum() + \
                  (args.beta/(self.sigma2*num_non_zero)).sum()
        # regularization
        l2_reg = 0
        for param in self.rnn_model.parameters():
          l2_reg += torch.norm(param)
        loss3 = args.network_reg*l2_reg

        loss = loss1+loss2+loss3
        loss.backward()
        nn.utils.clip_grad_norm_(self.rnn_model.parameters(), 5.0)
        #nn.utils.clip_grad_norm_(self.sigma2, 1.0)
        optimizer.step()
        # avoid numerical issues
        self.sigma2.data.clamp_(min=1e-6)

        if np.remainder(t,10) == 0:
          print('Iter {:d}  Training Loss:{:.4f}  Part1:{:.4f}  Part2:{:.4f}  Part3:{:.4f}'.format(
            t, float(loss.data), float(loss1.data), float(loss2.data), float(loss3.data)))
        train_loss.append(float(loss1.data)) # only save the likelihood part

  def predict(self, args, test_sequence):
    '''Predict test sequence labels using UISRNN model.

    Args:
      args: Model and testing configurations. See demo for description.

      test_sequence (real 2d numpy array, size: N by D): The test d_vector sequence.
        N: length of one test utterance
        D: observation dimension
        Example: 
          test_sequence= [[2.2 -1.0 3.0 5.6]    --> 1st entry of utterance 'iccc'
                          [0.5 1.8 -3.2 0.4]    --> 2nd entry of utterance 'iccc'
                          [-2.2 5.0 1.8 3.7]    --> 3rd entry of utterance 'iccc'
                          [-3.8 0.1 1.4 3.3]    --> 4th entry of utterance 'iccc'
                          [0.1 2.7 3.5 -1.7]]   --> 5th entry of utterance 'iccc'
          Here N=5, d=4.
    
    Returns:
      predict_speaker_id (a vector of integers, size: N): Predicted speaker id sequence.
      Example:
        predict_speaker_id = [0, 1, 0, 0, 1]
    '''
    test_sequence_length = test_sequence.shape[0]
    if args.model_type == 'generative':
      self.rnn_model.eval()
      test_sequence = np.tile(test_sequence, (args.test_iteration,1))
      test_sequence = Variable(torch.from_numpy(test_sequence).float())
      if torch.cuda.is_available():
        test_sequence = test_sequence.cuda()
      # bookkeeping for beam search
      proposal_set = [([],[],0,[],[])] # each cell consists of: (mean_set, hidden_set, score/-likelihood, trace, block_counts)
      max_speakers = 0

      for t in np.arange(0,args.test_iteration*test_sequence_length,args.look_ahead):
        l_remain = args.test_iteration*test_sequence_length-t
        score_set = float('inf')*np.ones(np.append(args.beam_size, max_speakers+1+np.arange(np.min([l_remain,args.look_ahead]))))
        for proposal_rank, proposal in enumerate(proposal_set):
          mean_buffer = list(proposal[0])
          hidden_buffer = list(proposal[1])
          score_buffer = proposal[2]
          trace_buffer = proposal[3]
          block_counts_buffer = list(proposal[4])
          n_speakers = len(mean_buffer)
          proposal_score_subset = float('inf')*np.ones(n_speakers+1+np.arange(np.min([l_remain,args.look_ahead])))
          for speaker_seq, _ in np.ndenumerate(proposal_score_subset):
            new_mean_buffer = mean_buffer.copy()
            new_hidden_buffer = hidden_buffer.copy()
            new_trace_buffer = trace_buffer.copy()
            new_block_counts_buffer = block_counts_buffer.copy()
            new_n_speakers = n_speakers
            new_loss = 0
            update_score = True
            for sub_idx, speaker in enumerate(speaker_seq):
              if speaker > new_n_speakers: # invalid trace
                update_score = False
                break
              if speaker < new_n_speakers: # existing speakers
                new_last_speaker = new_trace_buffer[-1]
                loss = weighted_mse_loss(torch.squeeze(new_mean_buffer[speaker]), test_sequence[t+sub_idx,:], 1/(2*self.sigma2)).cpu().detach().numpy()
                if speaker == new_last_speaker:
                  loss -= np.log(1-self.transition_bias)
                else:
                  loss -= np.log(self.transition_bias) + np.log(new_block_counts_buffer[speaker]) - np.log(sum(new_block_counts_buffer)+args.crp_theta)
                # update new mean and new hidden
                mean, hidden = self.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), new_hidden_buffer[speaker])
                # new_mean_buffer[speaker] = mean.clone()
                new_mean_buffer[speaker] = (new_mean_buffer[speaker]*((np.array(new_trace_buffer)==speaker).sum()-1).astype(float) + mean.clone())/\
                  (np.array(new_trace_buffer)==speaker).sum().astype(float)  # use mean to predict
                new_hidden_buffer[speaker] = hidden.clone()
                if speaker != new_trace_buffer[-1]:
                  new_block_counts_buffer[speaker] += 1
                new_trace_buffer.append(speaker)
              else: # new speaker
                init_input = Variable(torch.zeros(256)).unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                  init_input = init_input.cuda()
                mean, hidden = self.rnn_model(init_input, self.rnn_init_hidden.unsqueeze(0))
                loss = weighted_mse_loss(torch.squeeze(mean), test_sequence[t+sub_idx,:], 1/(2*self.sigma2)).cpu().detach().numpy()
                loss -= np.log(self.transition_bias) + np.log(args.crp_theta) - np.log(sum(new_block_counts_buffer)+args.crp_theta)
                # update new min and new hidden
                mean, hidden = self.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), hidden)
                new_mean_buffer.append(mean.clone())
                new_hidden_buffer.append(hidden.clone())
                new_block_counts_buffer.append(1)
                new_trace_buffer.append(speaker)
                new_n_speakers += 1
              new_loss += loss
            if update_score:
              score_set[tuple([proposal_rank])+speaker_seq] = score_buffer + new_loss

        # find top scores
        score_ranked = np.sort(score_set, axis=None)
        score_ranked[score_ranked==float('inf')] = 0
        score_ranked = np.trim_zeros(score_ranked)
        idx_ranked = np.argsort(score_set, axis=None)

        # update best traces
        new_proposal_set = []
        max_speakers = 0
        for new_proposal_rank in range(np.min((len(score_ranked), args.beam_size))):
          total_idx = np.unravel_index(idx_ranked[new_proposal_rank], score_set.shape)
          prev_proposal_idx = total_idx[0]
          new_speaker_idx = total_idx[1:]
          (mean_set, hidden_set, score, trace, block_counts) = proposal_set[prev_proposal_idx]
          new_mean_set = mean_set.copy()
          new_hidden_set = hidden_set.copy()
          new_score = score_ranked[new_proposal_rank] # can safely update the likelihood for now
          new_trace = trace.copy()
          new_block_counts = block_counts.copy()
          new_n_speakers = len(new_mean_set)
          max_speakers = max(max_speakers, new_n_speakers)
          for sub_idx, speaker in enumerate(new_speaker_idx): # update the proposal step-by-step
            if speaker == new_n_speakers:
              init_input = Variable(torch.zeros(args.toy_data_d_observation)).unsqueeze(0).unsqueeze(0)
              if torch.cuda.is_available():
                init_input = init_input.cuda()
              mean, hidden = self.rnn_model(init_input, self.rnn_init_hidden.unsqueeze(0))
              mean, hidden = self.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), hidden)
              new_mean_set.append(mean.clone())
              new_hidden_set.append(hidden.clone())
              new_block_counts.append(1)
              new_trace.append(speaker)
              new_n_speakers += 1
              max_speakers = max(max_speakers, new_n_speakers)
            else:
              mean, hidden = self.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), new_hidden_set[speaker])
              # new_mean_set[speaker] = mean.clone()
              new_mean_set[speaker] = (new_mean_set[speaker]*((np.array(new_trace)==speaker).sum()-1).astype(float) + mean.clone())/\
                (np.array(new_trace)==speaker).sum().astype(float) # use mean to predict
              new_hidden_set[speaker] = hidden.clone()
              if speaker != new_trace[-1]:
                new_block_counts[speaker] += 1
              new_trace.append(speaker)
          new_proposal_set.append((new_mean_set,new_hidden_set,new_score,new_trace,new_block_counts))
        proposal_set = new_proposal_set

      predict_speaker_id = proposal_set[0][3][-test_sequence_length:]
      return predict_speaker_id