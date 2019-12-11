# Contributor information:
#   Name: Xiang Lyu
#   GitHub account: aluminumbox
#   Email: aluminumbox@alumni.sjtu.edu.cn
#   Organization: Ping An Technology (Shanghai) Co., Ltd.
import unittest
import numpy as np

from uisrnn.contrib import get_prior_args


class TestGetPriorArgs(unittest.TestCase):
  """Tests for get_prior_args.py."""

  def test_get_k_t(self):
    """Test the return value of get_k_t(cluster_id_single)."""
    cluster_id_single = np.array(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]))
    k_t = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3])
    self.assertTrue((get_prior_args.get_k_t(cluster_id_single) == k_t).all())

  def test_get_n_kt(self):
    """Test the return value of get_n_kt(cluster_id_single)."""
    cluster_id_single = np.array(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]))
    n_kt = np.array(
      [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 1., 0.],
       [1., 1., 0.], [2., 1., 0.], [2., 1., 0.], [2., 1., 0.], [2., 1., 1.],
       [2., 1., 1.], [2., 1., 1.]])
    self.assertTrue((get_prior_args.get_n_kt(cluster_id_single) == n_kt).all())

  def test_get_normalized_id(self):
    """Test the return value of get_normalized_id(cluster_id_single)."""
    cluster_id_single = np.array(
      ['129_0', '129_0', '129_0', '129_1', '129_1', '129_1', '129_0', '129_0',
       '129_0', '129_2', '129_2', '129_2'])
    normalized_id = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2])
    self.assertTrue((get_prior_args.get_normalized_id(
      cluster_id_single) == normalized_id).all())


if __name__ == '__main__':
  unittest.main()
