import sys
import unittest

import torch.optim as optim

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import train_utils


class TrainUtilsTest(unittest.TestCase):
    def test_choose_optimizer_adam(self):
        result = train_utils.choose_optimizer("Adam")
        self.assertEqual(result, optim.Adam)


    def test_choose_optimizer_SGD(self):
        result = train_utils.choose_optimizer("SGD")
        self.assertEqual(result, optim.SGD)

    
    def test_choose_optimizer(self):
        got_error = False
        try:
            train_utils.choose_optimizer("1234567890-STRING")
        except NotImplementedError as error:
            got_error = True

        self.assertTrue(got_error)


if __name__ == '__main__':
    unittest.main()
