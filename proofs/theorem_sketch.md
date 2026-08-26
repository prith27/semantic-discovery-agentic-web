# Theorem sketch (to paste into paper §2)

## Setup
- Agents i = 1..N with true skills s_i ∈ {0,1}^d and ads ŝ_i ∈ {0,1}^d
- Task requirement r ∈ {0,1}^d
- Team C covers r under true skills iff ∨_{i∈C} s_i ≥ r (bitwise)
- Similarity top-k: argmax_{|C|=k} Σ_{i∈C} cos(r, ŝ_i)
- Complementary: find small C that covers r under ŝ (then evaluate under s)

## Proposition 1 (Complementarity gap)
There exist registries and r such that a complementary team of size k covers r,
but every similarity top-k team fails to cover r (even with ŝ = s).

Proof sketch: let d=2k, r = 1^{2k}. Give k specialist agents each holding a unique
pair of bits (disjoint cover). Give k near-duplicates of r that each miss a different
critical bit but have high cos(r, ·). Similarity prefers the near-duplicates; union misses bits.

## Proposition 2 (Capability spam)
If agents may set ŝ_i ≥ s_i (overclaim), then for any M there exists a spam configuration
where similarity success is 0 while an honest complementary cover of size ≤ k exists,
and the similarity regret (gap vs optimal cover) is ≥ M under suitable encodings.

Implication: open-network discovery needs coverage objectives + outcome-grounded trust,
not embeddings alone.
