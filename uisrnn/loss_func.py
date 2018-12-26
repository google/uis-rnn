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
"""Loss functions for training."""

import torch


def weighted_mse_loss(input_tensor, target_tensor, weight=1):
  """Compute weighted MSE loss.

  Note that we are doing weighted loss that only sum up over non-zero entries.

  Args:
    input_tensor: input tensor
    target_tensor: target tensor
    weight: weight tensor, in this case 1/sigma^2

  Returns:
    the weighted MSE loss
  """
  observation_dim = input_tensor.size()[-1]
  streched_tensor = ((input_tensor - target_tensor) ** 2).view(
      -1, observation_dim)
  entry_num = float(streched_tensor.size()[0])
  non_zero_entry_num = torch.sum(streched_tensor[:, 0] != 0).float()
  weighted_tensor = torch.mm(
      ((input_tensor - target_tensor)**2).view(-1, observation_dim),
      (torch.diag(weight.float().view(-1))))
  return torch.mean(
      weighted_tensor) * weight.nelement() * entry_num / non_zero_entry_num


def sigma2_prior_loss(num_non_zero, sigma_alpha, sigma_beta, sigma2):
  """Compute sigma2 prior loss.

  Args:
    num_non_zero: since rnn_truth is a collection of different length sequences
        padded with zeros to fit them into a tensor, we count the sum of
        'real lengths' of all sequences
    sigma_alpha: inverse gamma shape
    sigma_beta: inverse gamma scale
    sigma2: sigma squared

  Returns:
    the sigma2 prior loss
  """
  return ((2 * sigma_alpha + num_non_zero + 2) /
          (2 * num_non_zero) * torch.log(sigma2)).sum() + (
              sigma_beta / (sigma2 * num_non_zero)).sum()


def regularization_loss(params, weight):
  """Compute regularization loss.

  Args:
    params: iterable of all parameters
    weight: weight for the regularization term

  Returns:
    the regularization loss
  """
  l2_reg = 0
  for param in params:
    l2_reg += torch.norm(param)
  return weight * l2_reg
