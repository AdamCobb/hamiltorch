import unittest
import torch.nn as nn
import hamiltorch.util
import torch


class UtilTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_flatten_unflatten(self):
        model = nn.Linear(4, 4)
        flattened_params = hamiltorch.util.flatten(model)

        new_model = nn.Linear(4, 4)
        hamiltorch.util.unflatten(new_model, flattened_params)
        new_model_flattened_params = hamiltorch.util.flatten(new_model)

        hamiltorch.util.eval_print('flattened_params', 'new_model_flattened_params')

        self.assertTrue(torch.all(torch.eq(flattened_params, new_model_flattened_params)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
