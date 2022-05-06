import torch
import sys
import unittest

import pandas as pd 
import numpy as np

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import common_utils


class CommonUtilsTest(unittest.TestCase):
    def test_flip(self):
        input = torch.tensor([[[0, 1],
                               [2, 3]],
                              [[4, 5],
                               [6, 7]]])
        result = common_utils.flip(input)
        expected = torch.tensor([[[4, 5],
                                  [6, 7]],
                                 [[0, 1],
                                  [2, 3]]])
        self.assertTrue(torch.equal(result, expected))


    def test_min_clip_row(self):
        r"""
        Gets maximum along the rows. 
        """
        input = torch.tensor([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        result_1, result_2 = common_utils.min_clip(input, 1)
        expected_1 = torch.tensor([ 3, 7, 11, 15])
        expected_2 = torch.tensor([3, 3, 3, 3])
        self.assertTrue(torch.equal(result_1, expected_1))
        self.assertTrue(torch.equal(result_2, expected_2))


    def test_min_clip_col(self):
        r"""
        Gets maximum along the columns. 
        """
        input = torch.tensor([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        result_1, result_2 = common_utils.min_clip(input, 0)
        expected_1 = torch.tensor([ 12,  13, 14, 15])
        expected_2 = torch.tensor([3, 3, 3, 3])
        self.assertTrue(torch.equal(result_1, expected_1))
        self.assertTrue(torch.equal(result_2, expected_2))
  

    def test_max_clip_row(self):
        r"""
        Gets minimum along the rows. 
        """
        input = torch.tensor([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        result_1, result_2 = common_utils.max_clip(input, 1)
        expected_1 = torch.tensor([ 0, 4, 8, 12])
        expected_2 = torch.tensor([0, 0, 0, 0])
        self.assertTrue(torch.equal(result_1, expected_1))
        self.assertTrue(torch.equal(result_2, expected_2))


    def test_max_clip_col(self):
        r"""
        Gets minimum along the columns. 
        """
        input = torch.tensor([[0, 1, 2, 3],
                              [4, 5, 6, 7],
                              [8, 9, 10, 11],
                              [12, 13, 14, 15]])
        result_1, result_2 = common_utils.max_clip(input, 0)
        expected_1 = torch.tensor([ 0, 1, 2, 3])
        expected_2 = torch.tensor([0, 0, 0, 0])
        self.assertTrue(torch.equal(result_1, expected_1))
        self.assertTrue(torch.equal(result_2, expected_2))


    def test_normalize_0to1(self):
        input = torch.tensor([[1, 2, 3], 
                              [2, 3, 2], 
                              [3, 2, 1]])
        result = common_utils.normalize(input, 0, 1)
        expected = torch.tensor([[0.0000, 0.5000, 1.0000],
                                 [0.5000, 1.0000, 0.5000],
                                 [1.0000, 0.5000, 0.0000]])
        self.assertTrue(torch.equal(result, expected))


    def test_normalize_0to255(self):
        input = torch.tensor([[1, 2, 3], 
                              [2, 3, 2], 
                              [3, 2, 1]])
        result = common_utils.normalize(input, 0, 255)
        expected = torch.tensor([[0.0000, 127.5000, 255.0000],
                                 [127.5000, 255.0000, 127.5000],
                                 [255.0000, 127.5000, 0.0000]])
        self.assertTrue(torch.equal(result, expected))


    def test_pandas_col_to_numpy(self):
        df = pd.DataFrame(data={'col1': ['78.09', '5.01', '[1\n5.01  ]', '  52.01'], 'col2': ['1', '1', '1', '1']})
        df_col = df['col1']
        result = common_utils.pandas_col_to_numpy(df_col)
        expected = np.array([[78.09],[ 5.01], [15.01], [52.01]])
        self.assertEqual(result.all(), expected.all())


    def test_pandas_string_to_numpy(self):
        a = "[100\n1]10.00  "
        result = common_utils.pandas_string_to_numpy(a)
        expected = np.array([100110.00])
        self.assertEqual(result.all(), expected.all())


if __name__ == '__main__':
    unittest.main()
