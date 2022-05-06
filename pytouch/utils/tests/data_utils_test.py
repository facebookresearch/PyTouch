import torch
import sys
import unittest

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import data_utils


class DataUtilsTest(unittest.TestCase):
    def test_interpolate_img_same_dimension(self):
        img = torch.tensor([[[1., 2., 3.],
                             [1., 2., 3.],
                             [1., 2., 3.]]])

        result = data_utils.interpolate_img(img, 3, 3)
        expected = torch.tensor([[[1., 2., 3.],
                                  [1., 2., 3.],
                                  [1., 2., 3.]]])
        self.assertTrue(torch.equal(result, expected))

    def test_interpolate_img_higher_dimension(self):
        img = torch.tensor([[[1., 2., 3.],
                             [1., 2., 3.],
                             [1., 2., 3.]]])

        result = data_utils.interpolate_img(img, 6, 6)
        expected = torch.tensor([[[1., 1., 2., 2., 3., 3.],
                                  [1., 1., 2., 2., 3., 3.],
                                  [1., 1., 2., 2., 3., 3.],
                                  [1., 1., 2., 2., 3., 3.],
                                  [1., 1., 2., 2., 3., 3.],
                                  [1., 1., 2., 2., 3., 3.]]])
        self.assertTrue(torch.equal(result, expected))

    def test_interpolate_img_lower_dimension(self):
        img = torch.tensor([[[1., 1., 2., 2., 3., 3.],
                             [1., 1., 2., 2., 3., 3.],
                             [1., 1., 2., 2., 3., 3.],
                             [1., 1., 2., 2., 3., 3.],
                             [1., 1., 2., 2., 3., 3.],
                             [1., 1., 2., 2., 3., 3.]]])

        result = data_utils.interpolate_img(img, 3, 3)
        expected = torch.tensor([[[1., 2., 3.],
                                  [1., 2., 3.],
                                  [1., 2., 3.]]])
        self.assertTrue(torch.equal(result, expected))

    def test_interpolate_empty_img(self):
        try:
            img = torch.tensor([[[],
                                [],
                                [],
                                [],
                                [],
                                []]])

            data_utils.interpolate_img(img, 3, 3)
        except RuntimeError as error:
            return

        self.fail()


if __name__ == '__main__':
    unittest.main()
