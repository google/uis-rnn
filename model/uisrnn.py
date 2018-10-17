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

from model.utils import sample_permuted_segments, weighted_mse_loss


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
  def __init__(self, input_dim, rnn_hidden_size, rnn_depth, rnn_dropout, observation_dim, transition_bias, sigma2):
    if torch.cuda.is_available():
      self.rnn_model = NormalRNN(input_dim, rnn_hidden_size, rnn_depth, rnn_dropout, observation_dim).cuda()
      self.rnn_init_hidden = nn.Parameter(torch.zeros(1, rnn_hidden_size).cuda())
      if sigma2 == None:
        self.sigma2 = nn.Parameter(.1*torch.ones(observation_dim).cuda())
      else:
        self.sigma2 = nn.Parameter(sigma2*torch.ones(observation_dim).cuda())
    else:
      self.rnn_model = NormalRNN(input_dim, rnn_hidden_size, rnn_depth, rnn_dropout, observation_dim)
      self.rnn_init_hidden = nn.Parameter(torch.zeros(1, rnn_hidden_size))
      if sigma2 == None:
        self.sigma2 = nn.Parameter(.1*torch.ones(observation_dim))
      else:
        self.sigma2 = nn.Parameter(sigma2*torch.ones(observation_dim))
    self.transition_bias = transition_bias


def resize_seq(args, sequence, cluster_id):
  '''Resize sequences for packing and batching

  Args:
    sequence (real numpy matrix, size: seq_len*obs_size): observation sequence
    cluster_id (real vector, size: seq_len): cluster indicator sequence
  Returns:
    packed_rnn_input:
    rnn_truth:
    bias: flipping coin head probability
  '''

  obs_size = np.shape(sequence)[1]
  # merge sub-sequences that belong to a single cluster to a single sequence
  unique_id = np.unique(cluster_id)
  if args.permutation is None:
    num_clusters = len(unique_id)
  else:
    num_clusters = len(unique_id)*args.permutation
  sub_sequences = []
  seq_lengths = []
  if args.permutation is None:
    for i in unique_id:
      sub_sequences.append(sequence[np.where(cluster_id==i),:][0])
      seq_lengths.append(len(np.where(cluster_id==i)[0])+1)
  else:
    for i in unique_id:
      idx_set = np.where(cluster_id==i)[0]
      sampled_idx_sets = sample_permuted_segments(idx_set, args.permutation)
      for j in range(args.permutation):
        sub_sequences.append(sequence[sampled_idx_sets[j],:])
        seq_lengths.append(len(idx_set)+1)

  # compute bias
  transit_num = 0
  for entry in range(len(cluster_id)-1):
     transit_num += (cluster_id[entry]!=cluster_id[entry+1])
  # return sub_sequences, seq_lengths, transit_num/(len(cluster_id)-1)
  return sub_sequences, seq_lengths, 0.158


def pack_seq(rnn_input, sorted_seq_lengths):
  packed_rnn_input = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, sorted_seq_lengths, batch_first=False)
  # ground truth is the shifted input
  rnn_truth = rnn_input[1:,:,:]
  return packed_rnn_input, rnn_truth


def init_model(args, sequence, speaker_id):
  num_data, observation_dim = sequence.shape
  input_dim = observation_dim
  model = UISRNN(input_dim, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout, observation_dim, .5, args.sigma2)
  return model


def fit(args, train_sequence, train_cluster_id):
  if args.model_type == 'generative':
    num_data, observation_dim = train_sequence.shape
    input_dim = observation_dim
    model = UISRNN(input_dim, args.rnn_hidden_size, args.rnn_depth, args.rnn_dropout, observation_dim, .5, args.sigma2)
    model.rnn_model.train()

    if args.optimizer == 'adam':
      if args.sigma2 == None: # train sigma2
        optimizer = optim.Adam([
          {'params': model.rnn_model.parameters()}, # rnn parameters
          {'params': model.rnn_init_hidden}, # rnn initial hidden state
          {'params': model.sigma2} # variance parameters
          ], lr=args.learn_rate)
      else: # don't train sigma2
        optimizer = optim.Adam([
          {'params': model.rnn_model.parameters()}, # rnn parameters
          {'params': model.rnn_init_hidden} # rnn initial hidden state
          ], lr=args.learn_rate)

    sub_sequences, seq_lengths, transition_bias = resize_seq(args, train_sequence, train_cluster_id)
    num_clusters = len(seq_lengths)
    sorted_seq_lengths = np.sort(seq_lengths)[::-1]
    permute_index = np.argsort(seq_lengths)[::-1]
    model.transition_bias = transition_bias
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
        hidden = torch.mm(torch.ones(args.batch_size,1).float().cuda(), model.rnn_init_hidden).unsqueeze(0)
      else:
        hidden = torch.mm(torch.ones(args.batch_size,1).float(), model.rnn_init_hidden).unsqueeze(0)
      mean, _ = model.rnn_model(packed_train_sequence, hidden)
      # use mean to predict
      mean = torch.cumsum(mean, dim=0)
      mean_size = mean.size()
      if torch.cuda.is_available():
        mean = torch.mm(torch.diag(1.0/torch.arange(1,mean_size[0]+1).float().cuda()), mean.view(mean_size[0],-1))
      else:
        mean = torch.mm(torch.diag(1.0/torch.arange(1,mean_size[0]+1).float()), mean.view(mean_size[0],-1))
      mean = mean.view(mean_size)

      loss1 = weighted_mse_loss((rnn_truth!=0).float()*mean[:-1,:,:], rnn_truth, 1/(2*model.sigma2))
      weight = (((rnn_truth!=0).float()*mean[:-1,:,:] - rnn_truth)**2).view(-1,observation_dim)
      sum_weight = torch.sum(weight, dim=0).squeeze()
      num_non_zero = torch.sum((weight!=0).float(), dim=0).squeeze()
      loss2 = ((2*args.alpha+num_non_zero+2)/(2*num_non_zero)*torch.log(model.sigma2)).sum() + \
                (args.beta/(model.sigma2*num_non_zero)).sum()
      # regularization
      l2_reg = 0
      for param in model.rnn_model.parameters():
        l2_reg += torch.norm(param)
      loss3 = args.network_reg*l2_reg

      loss = loss1+loss2+loss3
      loss.backward()
      nn.utils.clip_grad_norm_(model.rnn_model.parameters(), 5.0)
      #nn.utils.clip_grad_norm_(model.sigma2, 1.0)
      optimizer.step()
      # avoid numerical issues
      model.sigma2.data.clamp_(min=1e-6)

      if np.remainder(t,10) == 0:
        print('Iter {:d}  Training Loss:{:.4f}  Part1:{:.4f}  Part2:{:.4f}  Part3:{:.4f}'.format(
          t, float(loss.data), float(loss1.data), float(loss2.data), float(loss3.data)))
      train_loss.append(float(loss1.data)) # only save the likelihood part

    return model


def predict(args, model, test_sequence, test_cluster_id):
  '''Model testing

  Predict cluster labels given the input test sequence

  Args:
    model:
    test_sequence:
    test_cluster_id:
  Returns:
    predicted sequence of cluster ids
  '''
  if args.model_type == 'generative':
    model.rnn_model.eval()
    test_sequence = np.tile(test_sequence, (args.test_iteration,1))
    test_sequence = Variable(torch.from_numpy(test_sequence).float())
    if torch.cuda.is_available():
      test_sequence = test_sequence.cuda()
    # bookkeeping for beam search
    proposal_set = [([],[],0,[],[])] # each cell consists of: (mean_set, hidden_set, score/-likelihood, trace, block_counts)
    max_speakers = 0

    for t in np.arange(0,args.test_iteration*len(test_cluster_id),args.look_ahead):
      l_remain = args.test_iteration*len(test_cluster_id)-t
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
              loss = weighted_mse_loss(torch.squeeze(new_mean_buffer[speaker]), test_sequence[t+sub_idx,:], 1/(2*model.sigma2)).cpu().detach().numpy()
              if speaker == new_last_speaker:
                loss -= np.log(1-model.transition_bias)
              else:
                loss -= np.log(model.transition_bias) + np.log(new_block_counts_buffer[speaker]) - np.log(sum(new_block_counts_buffer)+args.crp_theta)
              # update new mean and new hidden
              mean, hidden = model.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), new_hidden_buffer[speaker])
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
              mean, hidden = model.rnn_model(init_input, model.rnn_init_hidden.unsqueeze(0))
              loss = weighted_mse_loss(torch.squeeze(mean), test_sequence[t+sub_idx,:], 1/(2*model.sigma2)).cpu().detach().numpy()
              loss -= np.log(model.transition_bias) + np.log(args.crp_theta) - np.log(sum(new_block_counts_buffer)+args.crp_theta)
              # update new min and new hidden
              mean, hidden = model.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), hidden)
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
            mean, hidden = model.rnn_model(init_input, model.rnn_init_hidden.unsqueeze(0))
            mean, hidden = model.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), hidden)
            new_mean_set.append(mean.clone())
            new_hidden_set.append(hidden.clone())
            new_block_counts.append(1)
            new_trace.append(speaker)
            new_n_speakers += 1
            max_speakers = max(max_speakers, new_n_speakers)
          else:
            mean, hidden = model.rnn_model(test_sequence[t+sub_idx,:].unsqueeze(0).unsqueeze(0), new_hidden_set[speaker])
            # new_mean_set[speaker] = mean.clone()
            new_mean_set[speaker] = (new_mean_set[speaker]*((np.array(new_trace)==speaker).sum()-1).astype(float) + mean.clone())/\
              (np.array(new_trace)==speaker).sum().astype(float) # use mean to predict
            new_hidden_set[speaker] = hidden.clone()
            if speaker != new_trace[-1]:
              new_block_counts[speaker] += 1
            new_trace.append(speaker)
        new_proposal_set.append((new_mean_set,new_hidden_set,new_score,new_trace,new_block_counts))
      proposal_set = new_proposal_set
    return proposal_set[0][3][-len(test_cluster_id):]
