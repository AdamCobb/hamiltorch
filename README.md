# hamiltorch [![Build Status](https://travis-ci.com/AdamCobb/hamiltorch.svg?token=qJKqovbtw9EzCw99Nvg8&branch=master)](https://travis-ci.com/AdamCobb/hamiltorch)


 PyTorch-based library for Riemannian Manifold Hamiltonian Monte Carlo (RMHMC)
 
 * Perform HMC in user-defined log probabilities and in PyTorch neural networks (objects inheriting from the `torch.nn.Module`).
 * Available sampling schemes:
     * HMC
     * No-U-Turn Sampler
     * Implicit RMHMC
     * Explicit RMHMC

 ## How to install

```
pip install git+https://github.com/AdamCobb/hamiltorch
```

 ## How does it work?

 Please refer to my [blog post](https://adamcobb.github.io/journal/hamiltorch.html), or follow the [notebook-style tutorials](https://github.com/AdamCobb/hamiltorch/tree/master/notebooks).  

 ## How to cite?


Please consider citing the following paper if you use `hamiltorch` in your research:

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
