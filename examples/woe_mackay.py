"""
Description from MacKay's example
---------------------------------
This script recreates the Banburismus plaintext comparison example from
David J.C. MacKay's *Information Theory, Inference, and Learning Algorithms*
(Chapter 18.4 — "A Taste of Banburismus").

🧩 Concept of Turing's Banburismus
-----------------------------------
In World War II, Turing and his team developed statistical methods to detect
whether two encrypted messages might have come from *the same machine state*.
The example compares two plaintext English messages (u and v) and marks where
their letters match.

Each position is treated as a simple binary feature:
    match (1) or non-match (0)

If the letters match more often than chance, that's evidence the two messages
were enciphered by identical machine settings. If matches occur only at the
random rate, they're likely unrelated.

This is conceptually identical to how a **Naive Bayes model** or **credit
scorecard** works:
    - You have two hypotheses (Good/Bad, or H1/H0)
    - Each feature contributes independent log-odds evidence
    - Total score = sum of Weight of Evidence (WoE) across all features

🎲 Turing's Evidence Formula
-----------------------------------

Each match (“X”) or non-match (“0”) contributes a likelihood factor:

    Match:       β / (1/26)
    Non-match:   (1 - β) / (25/26)

Taking logs (in decibans) gives additive weights:

    WoE = 10 × log₁₀( P(z | H1) / P(z | H0) )

Turing's teams even used transparent overlays to sum decibans manually,
just like a paper scorecard.

📊 Binomial likelihood formulation
-----------------------------------
Turing modeled the repetition figure (the sequence of X and O symbols) as a
series of *independent Bernoulli trials* — each position being a binary event:

    X = match (success)
    O = non-match (failure)

If the messages really fit (same machine state), the probability of a match is θ,
where θ = ∑ pᵢ² ≈ 0.076 for English text.
If the messages do not fit (different machines), the probability of a match is 1/26
since any letter can appear equally often.

Hence, the probability of observing n matches in N positions follows a
*binomial distribution* under each hypothesis:

    P(n matches | fit right) = C(N, n) · θⁿ (1 - θ)^(N - n)
    P(n matches | fit wrong) = C(N, n) · (1/26)ⁿ (25/26)^(N - n)

The ratio of these two binomial likelihoods provides the evidence in favour
of a correct fit:

    Factor = [ P(n|fit right) / P(n|fit wrong) ]
            = (26θ)ⁿ · [ (26/25)(1 - θ) ]^(N - n)

The combinatorial term C(N, n) cancels out, leaving a simple product of per-letter
likelihood ratios.  Each X (match) contributes a factor of 26θ, and each O
(non-match) contributes a factor of (26/25)(1 - θ).

Taking 10·log₁₀ of this factor expresses the evidence in *decibans* — the same
Weight of Evidence scale later used in modern scorecards.  For English, with
θ = 0.076, each match contributes about +3.0 decibans and each non-match about -0.17 decibans.

💡 Credit Risk Theory Analogy
-----------------------------------
- Hypothesis H1 (“same machine”) ↔ “Good” (event of interest)
- Hypothesis H0 (“different machines”) ↔ “Bad” (baseline)
- Feature z_t = 1 if the two ciphertext letters match, else 0
- Each match contributes positive WoE, each non-match contributes slightly negative WoE
- Total log-odds (sum of WoE) gives the overall likelihood ratio between H1 and H0

The probability parameters correspond to class-conditional rates:
    P(z=1 | H1) = m = 0.076    (match probability in English)
    P(z=1 | H0) = 1/A = 1/26   (chance match if random)

Each observation's WoE is computed as:
    WoE = 10 * log10( P(z|H1) / P(z|H0) )

In this example:
    - Match → +3.1 decibans (strong evidence for H1)
    - Non-match → -0.18 decibans (slight evidence for H0)

The total evidence is the sum of all match/non-match contributions,
analogous to a model score in decibans (10 × log₁₀ odds).
"""

import math

from tabulate import tabulate


def print_evidence_decision_tree(u, v):
    """Show evidence accumulation as a decision tree."""
    A = 26
    m = 0.076
    woe_match = 10 * math.log10(m * A)
    woe_nonmatch = 10 * math.log10(((1 - m) * A) / (A - 1))

    cumulative = 0
    print("\n[Evidence Tree]\n")
    print("Start: 0.0 db (neutral prior)")

    for i, (char_u, char_v) in enumerate(zip(u, v, strict=False)):
        match = char_u == char_v
        woe = woe_match if match else woe_nonmatch
        cumulative += woe

        indent = "  " * min(i, 5)  # Limit indent
        symbol = "✓" if match else "✗"

        # Show key decision points
        if match or i < 10 or i % 15 == 0:
            print(
                f"{indent}├─ Pos {i}: {char_u}=={char_v}? {symbol} → {woe:+.2f}db → Total: {cumulative:+.1f}db"
            )

    odds = 10 ** (cumulative / 10)
    print(f"\n{'  ' * 5}└─ Final: {cumulative:+.1f} db → {odds:.0f}:1 in favor of H1")


def print_segmented_matches(u, v, segment_size=20):
    """Show matches in readable segments."""
    print("\n[Segmented Match Pattern]\n")

    for i in range(0, len(u), segment_size):
        u_seg = u[i : i + segment_size]
        v_seg = v[i : i + segment_size]
        matches = "".join(
            "*" if a == b else "." for a, b in zip(u_seg, v_seg, strict=False)
        )

        print(f"Position {i:>3}-{min(i + segment_size - 1, len(u) - 1):<3}")
        print(f"  u: {u_seg}")
        print(f"  v: {v_seg}")
        print(f"  ↓: {matches}")
        print()


# David J.C. MacKay's example (18.4)
# Two plaintexts
u = "LITTLE-JACK-HORNER-SAT-IN-THE-CORNER-EATING-A-CHRISTMAS-PIE--HE-PUT-IN-H"
v = "RIDE-A-COCK-HORSE-TO-BANBURY-CROSS-TO-SEE-A-FINE-LADY-UPON-A-WHITE-HORSE"

# Compute matches
MATCHES = "".join("*" if a == b else "." for a, b in zip(u, v, strict=False))
# Create table rows
table = [["u", u], ["v", v], ["matches:", MATCHES]]

print("\n[MacKay 18.4 'A taste of Banburismus']\n")

# Print table in book-like style
print(tabulate(table, tablefmt="outline", stralign="left"))

print_segmented_matches(u, v)
print_evidence_decision_tree(u, v)

# WoE computation
A = 26
m = 0.076

woe_match = 10 * math.log10(m / (1 / A))
woe_nonmatch = 10 * math.log10((1 - m) / (1 - 1 / A))

# Simplifying gives:
# # woe_match = 10 * math.log10(m * A)
# woe_nonmatch = 10 * math.log10(((1 - m) * A) / (A - 1))

M = MATCHES.count("*")
N = len(MATCHES) - M
total_woe = M * woe_match + N * woe_nonmatch

print(f"\nNumber of matches (M): {M}")
print(f"Number of non-matches (N): {N}")
print(f"WoE per match: {woe_match:.2f} decibans")
print(f"WoE per non-match: {woe_nonmatch:.2f} decibans")
print(f"Total WoE: {total_woe:.1f} decibans")
print(
    f"Posterior odds (approx): 10**({total_woe:.1f}/10) ≈ {10 ** (total_woe / 10):.0f}:1 in favor of H1"
)

# Turing's 2.3 example style (simple repeat theory)
# Suppose the repetition figure has N=72 letters with n=12 matches (X)
N = 72
n = 12
BETA = 0.076  # English match probability
A = 26

# Likelihood ratio for the entire repetition figure
LR = (26 * BETA) ** n * ((26 / 25) * (1 - BETA)) ** (N - n)
decibans_total = 10 * math.log10(LR)

print("\n[Turing 2.3 'Theory of repeats' calculation]\n")
print(f"Length of repetition figure N = {N}, matches n = {n}")
print(f"β (sum of p_i²) = {BETA}")
print(
    f"Factor for X = 26β = {26 * BETA:.3f}, factor for O = (26/25)(1−β) = {(26 / 25) * (1 - BETA):.3f}"
)
print(
    f"Total log-evidence ≈ {decibans_total:.1f} decibans (same as MacKay’s total WoE)"
)
