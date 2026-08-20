import unittest
import torch
import hamiltorch
from hamiltorch.hmc import HMC, RMHMC
from hamiltorch.symplectic import (
    SymplecticNeuralNetwork, GSymplecticNeuralNetwork, TimeSymmetricSymplectic,
)
from hamiltorch.experiment_utils import (banana_log_prob, funnel_log_prob,
                                        make_gp_regression_log_prob, normal_normal_conjugate,
                                        normalised_energy_distance)


class LeapfrogTrajectoryTestCase(unittest.TestCase):
    """HMC.step must return synchronized (q, p) pairs including the initial
    state, without changing the leapfrog proposal itself."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        self.sampler = HMC(step_size=0.1, L=5, log_prob_func=banana_log_prob, dim=2)
        self.q0 = torch.tensor([0., 100.])
        self.p0 = torch.tensor([0.3, -0.2])

    def _grad(self, q):
        q = q.detach().requires_grad_()
        return torch.autograd.grad(banana_log_prob(q), q)[0]

    def test_trajectory_shape_and_initial_state(self):
        qs, ps, gs = self.sampler.step(self.q0.clone(), self.p0.clone())
        self.assertEqual(qs.shape, (6, 2))
        self.assertEqual(ps.shape, (6, 2))
        self.assertEqual(gs.shape, (6, 2))
        self.assertTrue(torch.allclose(qs[0], self.q0))
        self.assertTrue(torch.allclose(ps[0], self.p0))

    def test_endpoint_matches_reference_leapfrog(self):
        qs, ps, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        q, p = self.q0.clone(), self.p0.clone()
        p = p + 0.05 * self._grad(q)
        for _ in range(5):
            q = q + 0.1 * p
            g = self._grad(q)
            p = p + 0.1 * g
        p = p - 0.05 * g
        self.assertTrue(torch.allclose(qs[-1], q, atol=1e-5))
        self.assertTrue(torch.allclose(ps[-1], p, atol=1e-5))

    def test_momentum_not_mutated_in_place(self):
        p0 = self.p0.clone()
        self.sampler.step(self.q0.clone(), p0)
        self.assertTrue(torch.allclose(p0, self.p0))

    def test_synchronized_pairs_conserve_hamiltonian(self):
        qs, ps, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        H = lambda q, p: -banana_log_prob(q) + 0.5 * (p ** 2).sum()
        hs = torch.stack([H(qs[i], ps[i]) for i in range(6)]).detach()
        self.assertLess(float((hs - hs[0]).abs().max()), 1e-3)


class TaoIntegratorTestCase(unittest.TestCase):
    """The explicit RMHMC integrator must be exactly reversible on the
    augmented state and approximately conserve the Riemannian Hamiltonian."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        torch.set_default_dtype(torch.float64)
        self.sampler = RMHMC(step_size=0.05, L=5, log_prob_func=banana_log_prob,
                             dim=2, softabs_const=10.)
        self.q0 = torch.tensor([0., 100.])
        # fixed momentum: the Gibbs draw's scale changed with the extended
        # target, and these tests are about the integrator, not the refresh
        self.p0 = torch.tensor([0.35, -0.20])

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _tao(self, q, p, q_cop, p_cop):
        r = self.sampler
        eps = r.step_size
        angle = torch.as_tensor(2. * r.binding_const * eps, dtype=q.dtype)
        c, s = torch.cos(angle), torch.sin(angle)
        for _ in range(r.L):
            dHdq, dHdp = r._dH(q, p_cop); p = p - .5 * eps * dHdq; q_cop = q_cop + .5 * eps * dHdp
            dHdq, dHdp = r._dH(q_cop, p); q = q + .5 * eps * dHdp; p_cop = p_cop - .5 * eps * dHdq
            qs_, qd_ = q + q_cop, q - q_cop
            ps_, pd_ = p + p_cop, p - p_cop
            q = .5 * (qs_ + c * qd_ + s * pd_); p = .5 * (ps_ - s * qd_ + c * pd_)
            q_cop = .5 * (qs_ - c * qd_ - s * pd_); p_cop = .5 * (ps_ + s * qd_ - c * pd_)
            dHdq, dHdp = r._dH(q_cop, p); q = q + .5 * eps * dHdp; p_cop = p_cop - .5 * eps * dHdq
            dHdq, dHdp = r._dH(q, p_cop); p = p - .5 * eps * dHdq; q_cop = q_cop + .5 * eps * dHdp
        return q, p, q_cop, p_cop

    def test_exact_augmented_reversibility(self):
        qf, pf, qcf, pcf = self._tao(self.q0.clone(), self.p0.clone(),
                                     self.q0.clone(), self.p0.clone())
        qb, _, _, _ = self._tao(qf, -pf, qcf, -pcf)
        self.assertLess(float((qb - self.q0).abs().max()), 1e-8)

    def test_step_matches_full_augmented_integration(self):
        qs, ps, _, _, _ = self.sampler.step(self.q0.clone(), self.p0.clone())
        qf, pf, _, _ = self._tao(self.q0.clone(), self.p0.clone(),
                                 self.q0.clone(), self.p0.clone())
        self.assertTrue(torch.allclose(qs[-1], qf, atol=1e-10))
        self.assertTrue(torch.allclose(ps[-1], pf, atol=1e-10))

    def test_hamiltonian_error_is_second_order(self):
        """|dH| falls ~4x per halving of eps once eps is small enough.

        Convergence is not monotone at moderate eps: Tao's binding rotation
        turns through 2*omega*eps per step, and near resonant angles the error
        stalls. At fixed T = 0.25, omega = 100 we measure
        [3.1e-2, 1.0e-3, 9.8e-4, 2.7e-4, 6.6e-5] for
        eps = 0.1, 0.05, 0.025, 0.0125, 0.00625 --- a 1.07x stall at 0.025
        followed by 3.6x and 4.0x. The order is therefore checked in the
        asymptotic regime, where the second-order rate is clean.
        """
        drifts = []
        for eps, L in [(0.025, 10), (0.0125, 20), (0.00625, 40)]:
            r = RMHMC(step_size=eps, L=L, log_prob_func=banana_log_prob,
                      dim=2, softabs_const=10.)
            qs, ps, _, _, _ = r.step(self.q0.clone(), self.p0.clone())
            drifts.append(abs(float(r.hamiltonian(qs[-1], ps[-1]).detach())
                              - float(r.hamiltonian(qs[0], ps[0]).detach())))
        self.assertLess(drifts[1] * 3, drifts[0])
        self.assertLess(drifts[2] * 3, drifts[1])


class SympNetInverseTestCase(unittest.TestCase):
    """SympNet blocks are shears with closed-form inverses; the time-symmetric
    composition Psi = (R Phi^{-1} R) Phi must be exactly momentum-reversible."""

    def setUp(self):
        hamiltorch.set_random_seed(3)
        torch.set_default_dtype(torch.float64)
        self.D = 2
        self.z = torch.randn(7, 2 * self.D)
        self.dt = torch.tensor(0.37)
        self.nets = [
            SymplecticNeuralNetwork(dim=2 * self.D, activation_modes=["up", "down"] * 4,
                                    channels=[8, 8] * 4),
            GSymplecticNeuralNetwork(dim=2 * self.D, activation_modes=["up", "down"],
                                     widths=[100, 100]),
        ]

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _flip(self, z):
        return torch.cat([z[..., :self.D], -z[..., self.D:]], -1)

    def test_inverse_round_trip(self):
        for net in self.nets:
            err = (net.inverse(net.step(self.z, self.dt), self.dt) - self.z).abs().max()
            self.assertLess(float(err), 1e-10)

    def test_time_symmetric_map_exactly_reversible(self):
        for net in self.nets:
            psi = TimeSymmetricSymplectic(net)
            fwd = psi.step(self.z, self.dt)
            back = psi.step(self._flip(fwd), self.dt)
            err = (self._flip(back) - self.z).abs().max()
            self.assertLess(float(err), 1e-10)


class RMHMCFieldStorageTestCase(unittest.TestCase):
    """RMHMC trajectories must carry the exact (dq/dt, dp/dt) field for
    surrogate gradient supervision."""

    def setUp(self):
        hamiltorch.set_random_seed(0)
        torch.set_default_dtype(torch.float64)
        self.sampler = RMHMC(step_size=0.05, L=5, log_prob_func=banana_log_prob,
                             dim=2, softabs_const=10.)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def test_fields_match_autograd_at_start(self):
        q0 = torch.tensor([0., 100.])
        p0, _ = self.sampler.gibbs(q0)
        _, _, fields, _, _ = self.sampler.step(q0, p0)
        self.assertEqual(fields.shape, (6, 4))
        qg, pg = q0.detach().requires_grad_(), p0.detach().requires_grad_()
        H = self.sampler.hamiltonian(qg, pg)
        dHdq, dHdp = torch.autograd.grad(H, (qg, pg))
        self.assertTrue(torch.allclose(fields[0], torch.cat([dHdp, -dHdq], -1), atol=1e-10))


class NewTargetsTestCase(unittest.TestCase):
    """Funnel and GP-regression log probs: finite values, gradients, and
    batch evaluation summing over rows."""

    def test_targets(self):
        for lp, w in [(funnel_log_prob, torch.tensor([0.5, 1.0])),
                      (make_gp_regression_log_prob(30), torch.zeros(3))]:
            wg = w.clone().requires_grad_()
            v = lp(wg)
            grad = torch.autograd.grad(v, wg)[0]
            self.assertTrue(torch.isfinite(v))
            self.assertTrue(torch.isfinite(grad).all())
            vb = lp(w[None, :].repeat(4, 1))
            self.assertLess(abs(float(vb) - 4 * float(v)), 1e-4)


class ExtendedRMHMCTestCase(unittest.TestCase):
    """The RMHMC proposal must satisfy both Metropolis conditions on the
    *extended* state, and the chain must recover a known target.

    Projecting to (q, p) and re-initialising the copies each iteration --- the
    earlier behaviour --- broke involutivity (4.5e-5) and, because the extended
    target's q-marginal has precision 2*Sigma^-1, contracted every standard
    deviation by 1/sqrt(2). Both the Metropolis ratio and the Gibbs refresh
    must use exp(-Hbar/2); halving only one leaves the two steps targeting
    different distributions and the contraction persists.
    """

    def setUp(self):
        hamiltorch.set_random_seed(0)
        torch.set_default_dtype(torch.float64)
        self.D = 2
        # normal-gamma rather than the banana: the banana's state magnitude
        # (~100) and stiffness make a central difference of the Tao map
        # numerically hopeless, which measures the probe and not the map
        self.sampler = RMHMC(step_size=0.05, L=5, log_prob_func=normal_normal_conjugate,
                             dim=self.D, softabs_const=10.)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _T(self, q, p, qb, pb):
        tq, tp, _, qbn, pbn = self.sampler.step(q, p, qb, pb)
        return tq[-1], tp[-1], qbn, pbn

    def test_involutive_on_extended_state(self):
        q0 = torch.ones(self.D)
        p0, pb0 = self.sampler.gibbs(q0, q0)
        flip = lambda q, p, qb, pb: (q, -p, qb, -pb)
        q1, p1, qb1, pb1 = self._T(q0, p0, q0, pb0)
        q2, p2, qb2, pb2 = self._T(*flip(q1, p1, qb1, pb1))
        back = flip(q2, p2, qb2, pb2)
        err = max(float((back[0] - q0).abs().max()), float((back[1] - p0).abs().max()),
                  float((back[2] - q0).abs().max()), float((back[3] - pb0).abs().max()))
        self.assertLess(err, 1e-10)

    def test_volume_preserving_on_extended_state(self):
        D = self.D
        q0 = torch.ones(D)
        p0, pb0 = self.sampler.gibbs(q0, q0)
        def flat(z):
            a, b, c, d = self._T(z[:D], z[D:2*D], z[2*D:3*D], z[3*D:])
            return torch.cat([a, b, c, d])
        # step() detaches inside _dH, so autograd reports a degenerate
        # Jacobian here; finite differences measure the real map
        z = torch.cat([q0, p0, q0, pb0]); h = 1e-6
        J = torch.zeros(4 * D, 4 * D)
        for i in range(4 * D):
            e = torch.zeros(4 * D); e[i] = h
            J[:, i] = (flat(z + e) - flat(z - e)) / (2 * h)
        self.assertLess(abs(float(torch.det(J).abs()) - 1.0), 1e-6)

    def test_recovers_a_known_gaussian(self):
        true_sd = torch.tensor([0.5, 1.0, 2.0])
        def gauss(w):
            return torch.distributions.MultivariateNormal(
                torch.zeros(3), torch.diag(true_sd ** 2)).log_prob(w).sum()
        hamiltorch.set_random_seed(2)
        r = RMHMC(step_size=0.15, L=10, log_prob_func=gauss, dim=3, softabs_const=1e6)
        traj, _, _, _ = r.sample(torch.zeros(3), num_samples=1200)
        sd = traj[:, -1, :].detach()[200:].std(0)
        # a chain targeting exp(-Hbar) instead would land near true_sd/sqrt(2)
        self.assertLess(float((sd - true_sd).abs().max()), 0.15)


class DistributionErrorTestCase(unittest.TestCase):
    """The correctness metric must separate distributions that ESS cannot.

    A map with a large involution defect yields weakly autocorrelated draws, so
    ESS reads high while the samples come from the wrong distribution. The
    energy distance is the check that catches it.
    """

    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(0)
        self.sd = torch.tensor([0.5, 1.0, 2.0])
        self.ref = torch.randn(1500, 3) * self.sd

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def _draw(self, n=1500):
        return torch.randn(n, 3) * self.sd

    def test_zero_for_the_same_distribution(self):
        self.assertLess(normalised_energy_distance(self._draw(), self.ref), 0.02)

    def test_detects_the_contracted_marginal(self):
        # the 1/sqrt(2) contraction produced by targeting exp(-Hbar) instead
        # of exp(-Hbar/2) in the extended RMHMC formulation
        contracted = self._draw() / (2 ** 0.5)
        self.assertGreater(normalised_energy_distance(contracted, self.ref), 0.015)

    def test_detects_the_split_support_artifact(self):
        # what an unsymmetrized SympNet actually produces: two lobes, hole at the mode
        bimodal = torch.cat([self._draw(750) - 2.0, self._draw(750) + 2.0])
        self.assertGreater(normalised_energy_distance(bimodal, self.ref), 0.2)


if __name__ == "__main__":
    unittest.main()
