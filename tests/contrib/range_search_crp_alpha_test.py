# Contributor information:
#   Name: Xiang Lyu
#   GitHub account: aluminumbox
#   Email: aluminumbox@alumni.sjtu.edu.cn
#   Organization: Ping An Technology (Shanghai) Co., Ltd.
"""This is the unit test module for uisrnn/contrib/range_search_crp_alpha.py."""
import unittest
import numpy as np

from uisrnn.contrib import range_search_crp_alpha


class TestRangeSearchCrpAlpha(unittest.TestCase):
  """Tests for range_search_crp_alpha.py."""

  def test_estimate_crp_alpha(self):
    """Test the return value of test_estimate_crp_alpha(train_cluster_id,
    search_range, search_step)."""
    train_cluster_id = np.array(
        ['0_0', '0_0', '0_1', '0_1', '0_1', '0_0', '0_0', '1_0', '1_0', '1_0',
         '1_1', '1_1', '1_1', '1_0', '1_0', '1_0', '1_2', '1_2', '1_2'])
    self.assertEqual(
        range_search_crp_alpha.estimate_crp_alpha(train_cluster_id, 1, 0.01),
        0.5)

  def test_get_k_t(self):
    """Test the return value of get_k_t(cluster_id_single)."""
    cluster_id_single = np.array(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]))
    k_t = np.array([1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3])
    self.assertTrue(
        (range_search_crp_alpha._get_k_t(cluster_id_single) == k_t).all())

  def test_get_n_kt(self):
    """Test the return value of get_n_kt(cluster_id_single)."""
    cluster_id_single = np.array(np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2]))
    n_kt = np.array(
        [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [1., 1., 0.], [1., 1., 0.],
         [1., 1., 0.], [2., 1., 0.], [2., 1., 0.], [2., 1., 0.], [2., 1., 1.],
         [2., 1., 1.], [2., 1., 1.]])
    self.assertTrue(
        (range_search_crp_alpha._get_n_kt(cluster_id_single) == n_kt).all())

  def test_get_normalized_id(self):
    """Test the return value of get_normalized_id(cluster_id_single)."""
    cluster_id_single = np.array(
        ['129_0', '129_0', '129_0', '129_1', '129_1', '129_1', '129_0', '129_0',
         '129_0', '129_2', '129_2', '129_2'])
    normalized_id = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2])
    self.assertTrue((range_search_crp_alpha._get_normalized_id(
        cluster_id_single) == normalized_id).all())


if __name__ == '__main__':
  unittest.main()
