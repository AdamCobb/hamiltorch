# hamiltorch [![Build Status](https://travis-ci.com/AdamCobb/hamiltorch.svg?token=qJKqovbtw9EzCw99Nvg8&branch=master)](https://travis-ci.com/AdamCobb/hamiltorch)


 PyTorch-based library for Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) and inference in Bayesian neural networks

 * Perform HMC in user-defined log probabilities and in PyTorch neural networks (objects inheriting from the `torch.nn.Module`).
 * Available sampling schemes:
     * HMC
     * No-U-Turn Sampler (currently adapts step-size only)
     * Implicit RMHMC
     * Explicit RMHMC
     * Symmetric Split HMC

 ## How to install

```
pip install git+https://github.com/AdamCobb/hamiltorch
```

 ## How does it work?

 There are currently two blog posts that describe how to use `hamiltorch`:

 * For basic usage and an introduction please refer to my earlier post in 2019 ["hamiltorch: a PyTorch Python package for sampling"](https://adamcobb.github.io/journal/hamiltorch.html)
 * For a more recent summary and a focus on Bayesian neural networks, please see my post ["Scaling HMC to larger data sets"](https://adamcobb.github.io/journal/bnn.html)

 There are also notebook-style tutorials:

 * [Sampling from generic log probabilities](https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_log_prob_examples.ipynb)
 * [Sampling from `torch.nn.Module` (basic)](https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_Bayesian_NN_example.ipynb)
 * [Bayesian neural networks and split HMC](https://github.com/AdamCobb/hamiltorch/blob/master/notebooks/hamiltorch_split_HMC_BNN_example.ipynb)

 ## How to cite?

Please consider citing the following papers if you use `hamiltorch` in your research:

For symmetric splitting:
```
@article{cobb2019introducing,
  title={Scaling Hamiltonian Monte Carlo Inference for Bayesian Neural Networks with Symmetric Splitting},
  author={Cobb, Adam D and Jalaian, Brian},
  journal={arXiv preprint arXiv:2010.06772},
  year={2020}
}
```

For RMHMC:
```
@article{cobb2019introducing,
  title={Introducing an Explicit Symplectic Integration Scheme for Riemannian Manifold Hamiltonian Monte Carlo},
  author={Cobb, Adam D and Baydin, At{\i}l{\i}m G{\"u}ne{\c{s}} and Markham, Andrew and Roberts, Stephen J},
  journal={arXiv preprint arXiv:1910.06243},
  year={2019}
}
```

 ## Who developed hamiltorch?

 [Adam D Cobb](https://adamcobb.github.io)

 [Atılım Güneş Baydin](http://www.robots.ox.ac.uk/~gunes/)

 [Brian Jalaian](https://www.brianjalaian.com)
