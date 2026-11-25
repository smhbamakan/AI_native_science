"""Figure 1: the Knowledge Delta Protocol as a layer beneath document-based publication (schematic)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
STY = {"doc": ("#ffffff", "#52514e"), "leg": ("#f3f2ee", "#52514e"), "art": ("#eef4fc", "#2a78d6"),
       "abs": ("#fff7e5", "#eda100"), "log": ("#eef4fc", "#2a78d6"), "lens": ("#fdf0eb", "#eb6834"),
       "view": ("#e8f6f0", "#1baf7a"), "agent": ("#ffffff", "#52514e")}

fig, ax = plt.subplots(figsize=(7.0, 8.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x0, y0, w, h, text, kind, size=9, bold_first=False):
    fc, ec = STY[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
    lines = text.split("\n")
    if bold_first:
        ax.text(x0 + w / 2, y0 + h / 2, "\n".join(lines), ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)
    else:
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35)
    return (x0, y0, w, h)


def arrow(p, q, dashed=False, color=INK2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 ls=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# column headers
for x, t in ((12, "Document-based\npublication"), (51, "Knowledge Delta Protocol"), (88, "Agents\n(human or AI)")):
    ax.text(x, 97, t, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
ax.plot([25, 25], [-14, 93], color="#dcdbd6", lw=1)
ax.plot([77, 77], [-14, 93], color="#dcdbd6", lw=1)

# left column
ys = [80, 62, 44, 26, 8]
txt = ["Research\n(branching,\niterative)", "Narrative\nmanuscript\n(PDF)", "One-off\npeer review", "Untyped\ncitations", "Citation-count\ncredit"]
for y, t in zip(ys, txt):
    box(2, y, 20, 11, t, "doc")
for y1, y2 in zip(ys[:-1], ys[1:]):
    arrow((12, y1), (12, y2 + 11))

# centre column
box(28, 80, 22, 11, "Legacy literature\n(prior\ndocuments)", "leg", 8.5)
box(53, 80, 22, 11, "Research artifacts\n(data, code,\nprotocols)", "art", 8.5)
box(28, 61, 22, 14, "Absorption profile\nα = (k, τ, γ, θ, a₀)\ndiscounted legacy\nevidence", "abs", 8.5)
box(28, 38, 47, 17, "Append-only log of signed transactions\nΔ = ⟨op, c, σ′, e, a, t, s⟩\nadmission: ≥ m independent attestations\nmemory decay with half-life τ", "log", 8.5)
box(28, 15, 47, 17, "Lenses λ: public, versioned\naggregation rules\nBayesian · replication-weighted\nargumentation · scope-aware\ndisagreement flags possible overstatement", "lens", 8)
for x, t in ((28, "Related work\n(computed)"), (44, "α-robust\nimpact"), (60, "Shapley\ncredit")):
    box(x, 0, 15, 10, t, "view", 8.5)
box(36, -13, 15, 10, "Frontier\nmap", "view", 8.5); box(52, -13, 15, 10, "Manuscript\nview", "view", 8.5)
arrow((39, 80), (39, 75)); arrow((39, 61), (39, 55)); arrow((64, 80), (64, 55))
arrow((51.5, 38), (51.5, 32)); arrow((51.5, 15), (51.5, 10))

# right column (agents)
box(79, 72, 19, 15, "Explore\n(swarm move;\nattractors\np, g, h)", "agent", 8.5)
box(79, 50, 19, 11, "Attest\n(replicate)", "agent", 8.5)
box(79, 24, 19, 16, "Re-explore\nwhen\nattestations fall\nbelow the record", "agent", 8.5)
box(79, 0, 19, 13, "Compile and\nrender views\n(artifacts → Δ)", "agent", 8.5)
arrow((88.5, 72), (88.5, 61))
arrow((79, 55), (75, 48))
arrow((79, 6), (75, 5))
FB = "#eb6834"
arrow((50, 68), (79, 79), dashed=True, color=FB, rad=-0.1)
arrow((75, 30), (79, 73), dashed=True, color=FB)
arrow((75, 22), (79, 30), dashed=True, color=FB)

ax.plot([2, 8], [-19, -19], color=INK2, lw=1.2); ax.text(9, -19, "content flow (evidence, transactions, computed views)", va="center", fontsize=8.5)
ax.plot([2, 8], [-24, -24], color=FB, lw=1.2, ls=(0, (4, 3))); ax.text(9, -24, "feedback that steers exploration (attractors h and g, anomaly signal)", va="center", fontsize=8.5)
ax.set_ylim(-27, 101)
fig.savefig("fig1_protocol.png", dpi=300, bbox_inches="tight")
print("ok")
