"""Minimal, replayable prototype of the Knowledge Delta Protocol (revision, reviewer request).

It contains: a JSON-serializable transaction schema, the eight-transaction log of Box 2 (hypothesis H17),
two lenses (a Bayesian log-odds lens with dependence clusters and a replication-weighted shrinkage lens),
retraction and qualification semantics, leave-one-out epistemic impact (equation (6)) and exact Shapley
credit over agents (equation (7)). Signatures are placeholders; nothing here is cryptographically secure.

Run: python3 kdp_prototype.py   -> prints the state trajectories used in Box 2 and writes kdp_example_log.json
"""
import json, math, itertools
from dataclasses import dataclass, asdict, field
from typing import Optional

OPS = {"assert", "support", "weaken", "qualify", "refute", "replicate", "retract", "equate"}
KAPPA = {"assert": 0.5, "support": 1.0, "replicate": 1.0, "weaken": -1.0, "refute": -2.0, "qualify": 1.0}


@dataclass
class Tx:
    id: str
    op: str
    claim: str                    # target claim id (for retract: the claim of the retracted transaction)
    scope: str                    # sigma': the scope the operation applies to ("all", "C", "notC")
    agent: str
    t: int
    evidence: dict = field(default_factory=dict)   # {"uri":..., "quality": q in [0,1], "cluster": dependence cluster}
    target: Optional[str] = None  # retract: id of retracted transaction; qualify: id of the new child claim
    sig: str = "placeholder"

    def __post_init__(self):
        assert self.op in OPS, self.op
        if self.op in ("retract", "qualify"): assert self.target


# scopes form a partial order: C and notC are both narrower than all
NARROWER = {("C", "all"), ("notC", "all")}
def within(s1, s2): return s1 == s2 or (s1, s2) in NARROWER


LOG = [
    Tx("d1", "assert", "H17", "C", "lab_A", 1, {"uri": "doi:example/1", "quality": 0.8, "cluster": "D1"}),
    Tx("d2", "support", "H17", "C", "lab_B", 2, {"uri": "doi:example/2", "quality": 0.7, "cluster": "D2"}),
    Tx("d3", "weaken", "H17", "notC", "lab_C", 3, {"uri": "doi:example/3", "quality": 0.8, "cluster": "D3"}),
    Tx("d4", "qualify", "H17", "C", "lab_D", 4, {"uri": "doi:example/4", "quality": 0.8, "cluster": "D4"}, target="H17|C"),
    Tx("d5", "replicate", "H17|C", "C", "lab_E", 5, {"uri": "doi:example/5", "quality": 0.9, "cluster": "D5"}),
    Tx("d6", "replicate", "H17|C", "C", "lab_F", 6, {"uri": "doi:example/6", "quality": 0.9, "cluster": "D6"}),
    Tx("d7", "weaken", "H17", "notC", "lab_G", 7, {"uri": "doi:example/7", "quality": 0.8, "cluster": "D1"}),   # reanalysis of D1: dependent on d1
    Tx("d8", "retract", "H17", "C", "lab_B", 8, {}, target="d2"),                                              # lab B retracts d2 after a coding error
]
CLAIMS = {"H17": "all", "H17|C": "C"}


def effective(log):
    """Retraction removes the evidential effect of its target; both stay in the audit log."""
    gone = {d.target for d in log if d.op == "retract"}
    return [d for d in log if d.id not in gone and d.op != "retract"]


def evidence_for(log, claim):
    """Transactions that bear on a claim: those on the claim itself, plus, for a qualified child claim, those on its
    parent whose operation scope lies within the child's scope. If the qualifying transaction is absent, the child
    does not exist and transactions on it are evaluated against the parent (scope lifting)."""
    eff = effective(log)
    children = {d.target: d for d in eff if d.op == "qualify"}
    if claim in children:
        sc = CLAIMS[claim]; parent = children[claim].claim
        return [d for d in eff if d.claim == claim or (d.claim == parent and within(d.scope, sc))]
    if claim in CLAIMS and claim != "H17":             # child whose qualifying transaction was removed
        return evidence_for(log, "H17")
    lifted = [d for d in eff if d.claim not in children and d.claim != claim and d.claim in CLAIMS]
    # a qualification is evidence for the narrower child claim, not for the unrestricted parent
    return [d for d in eff if d.claim == claim and d.op != "qualify"] + lifted


def bayes_lens(log, claim, l0=0.0):
    """Equation (3) with dependence clusters: each cluster contributes the mean signed strength of its transactions."""
    clusters = {}
    for d in evidence_for(log, claim):
        clusters.setdefault(d.evidence["cluster"], []).append(KAPPA[d.op] * d.evidence["quality"])
    l = l0 + sum(sum(v) / len(v) for v in clusters.values())
    return 1 / (1 + math.exp(-l))


def replication_lens(log, claim, mu0=0.5, n0=2.0):
    """One vote per independent dependence cluster: 1 if the cluster's net strength is positive, else 0.
    Replications count with weight 1, other operations with weight 0.5 (shrinkage towards mu0)."""
    clusters = {}
    for d in evidence_for(log, claim):
        clusters.setdefault(d.evidence["cluster"], []).append(d)
    num, den = n0 * mu0, n0
    for ds in clusters.values():
        net = sum(KAPPA[d.op] * d.evidence["quality"] for d in ds)
        w = 1.0 if any(d.op == "replicate" for d in ds) else 0.5
        num += w * (1.0 if net > 0 else 0.0); den += w
    return num / den


LENSES = {"Bayesian": bayes_lens, "replication-weighted": replication_lens}


def state(log, lens):
    return {c: LENSES[lens](log, c) for c in CLAIMS}


def impact(log, tx_id, lens, pi=None):
    """Equation (6): sum_c pi(c) |S^L(c) - S^{L without tx}(c)|, pi uniform by default."""
    pi = pi or {c: 1 / len(CLAIMS) for c in CLAIMS}
    a, b = state(log, lens), state([d for d in log if d.id != tx_id], lens)
    return sum(pi[c] * abs(a[c] - b[c]) for c in CLAIMS)


def shapley(log, lens, prior=0.5):
    """Equation (7): exact Shapley values over agents; v(Q) = D(S^{L_Q} || S^{L_empty}), uniform pi."""
    agents = sorted({d.agent for d in log}); n = len(agents)
    def v(Q):
        L = [d for d in log if d.agent in Q]
        s0 = {c: (prior if lens == "Bayesian" else 0.5) for c in CLAIMS}
        s = state(L, lens) if L else s0
        return sum(abs(s[c] - s0[c]) for c in CLAIMS) / len(CLAIMS)
    phi = {}
    for i in agents:
        others = [a for a in agents if a != i]; tot = 0.0
        for r in range(n):
            for Q in itertools.combinations(others, r):
                wgt = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                tot += wgt * (v(set(Q) | {i}) - v(set(Q)))
        phi[i] = tot
    return phi


if __name__ == "__main__":
    json.dump([asdict(d) for d in LOG], open("kdp_example_log.json", "w"), indent=1)
    print("step  op         " + "  ".join(f"{l[:5]}:{c:6s}" for l in LENSES for c in CLAIMS))
    for k in range(1, len(LOG) + 1):
        L = LOG[:k]
        born = any(d.op == "qualify" for d in L)
        vals = "  ".join((f"{state(L, l)[c]:.2f}" if (c == "H17" or born) else "-").rjust(12) for l in LENSES for c in CLAIMS)
        print(f"{LOG[k-1].id:5s} {LOG[k-1].op:10s} {vals}")
    for l in LENSES:
        print(l, "impact:", {d.id: round(impact(LOG, d.id, l), 3) for d in LOG})
        print(l, "Shapley:", {a: round(x, 3) for a, x in shapley(LOG, l).items()})
