import unittest
from hamiltorch.util import eval_print


class UtilTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_flatten_unflatten(self):
        a = True
        eval_print('a')

        self.assertTrue(a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
