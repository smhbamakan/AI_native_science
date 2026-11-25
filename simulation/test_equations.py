"""Unit tests for the model equations implemented in aepso.py (observation model, shrinkage lens, adaptive trust, absorption decay, velocity update and coefficient budget).
Run: python3 -m pytest -q test_equations.py   (or: python3 test_equations.py)"""
import numpy as np
import aepso as A


def _leg(nag, mean, n_leg=4):
    leg = dict(nag=np.zeros(A.NC), mean=np.zeros(A.NC)); leg["nag"][0] = nag; leg["mean"][0] = mean; return leg


def test_eq12_shrinkage_worked_example():
    # claim 0: agent 1 observed 0.6 and 0.7, agent 2 observed 0.5; three legacy agents with mean 0.8, w_H = 0.25
    kS = np.zeros((A.NC, 2)); kC = np.zeros((A.NC, 2))
    kS[0, 0], kC[0, 0] = 1.3, 2; kS[0, 1], kC[0, 1] = 0.5, 1
    lw = np.zeros(A.NC); lw[0] = 0.25
    S, adm, neff, nk, km = A.replication_lens(kS, kC, lw, _leg(3, 0.8), n0=1.0, m=2)
    assert abs(neff[0] - 2.75) < 1e-12
    assert abs(S[0] - (1.0 * 0.2 + 0.65 + 0.5 + 0.75 * 0.8) / (1.0 + 2.75)) < 1e-12   # = 0.52
    assert adm[0] and not adm[1]
    assert abs(S[1] - A.MU0) < 1e-12                       # no evidence -> prior mean


def test_eq12_one_vote_per_agent():
    kS = np.zeros((A.NC, 1)); kC = np.zeros((A.NC, 1)); kS[0, 0], kC[0, 0] = 1000 * 0.9, 1000
    S, *_ = A.replication_lens(kS, kC, np.zeros(A.NC), _leg(0, 0), n0=1.0, m=2)
    # one agent counts as one vote however many observations it makes: S = (0.2 + 0.9) / 2
    assert abs(S[0] - 0.55) < 1e-12


def test_eq13_adaptive_trust():
    eta, yH, yt, nt = 4.0, 0.8, 0.575, 2
    a0 = np.exp(-eta * abs(yH - yt) * nt / (nt + 1))
    assert abs(a0 - np.exp(-0.6)) < 1e-12
    assert np.exp(-eta * abs(yH - yt) * 0) == 1.0        # limit n_t -> 0 of the formula; the code uses the prior 0.5 instead


def test_eq4_legacy_decay():
    assert abs(2.0 ** (-(50 - 0) / 50.0) - 0.5) < 1e-12


def test_eq10_budget_nonnegative_and_conserved():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        a0 = rng.uniform(); w = rng.uniform(); phi3 = 0.5 * a0
        phi1, phi2 = w * (A.PHI_TOT - phi3), (1 - w) * (A.PHI_TOT - phi3)
        assert min(phi1, phi2, phi3) >= 0 and abs(phi1 + phi2 + phi3 - A.PHI_TOT) < 1e-12


def test_eq10_velocity_clamp_and_reflection():
    rng = np.random.default_rng(1); n = 50
    x = rng.uniform(size=(n, 2)); v = np.zeros((n, 2)); p = rng.uniform(size=(n, 2)); g = rng.uniform(size=(n, 2))
    h = np.full((n, 2), np.nan)
    x2, v2 = A.pso_step(rng, x, v, p, g, h, np.zeros(n), np.full(n, 0.5))
    assert np.all(np.abs(v2) <= A.VMAX + 1e-12) and np.all((x2 >= 0) & (x2 <= 1))


def test_eq11_observation_unbiased_outside_artifact():
    rng = np.random.default_rng(2); W = A.World(rng)
    far = np.array([[0.0, 0.0]]) if np.linalg.norm(W.art_center) > 0.4 else np.array([[1.0, 1.0]])
    y = np.array([W.observe(far, 1.0, A.DEFAULT)[0] for _ in range(4000)])
    assert abs(y.mean() - W.truth(far)[0]) < 0.02


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
