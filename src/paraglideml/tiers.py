"""
Flight-quality tier definitions — the single source of truth, kept dependency-free.

Lives in its own tiny module (no torch / sklearn / pandas) so the lean inference path
(`paraglideml[inference]`) and the training path can both import it without dragging
the heavy NN stack in via ordinal.py. Cumulative distance tiers: each implies the ones
below, so P(>=flyable) >= P(>=good) >= P(>=epic).
"""

from typing import List, Tuple

# (threshold_km, human label). Cumulative: each tier implies the tiers below it.
TIERS: List[Tuple[float, str]] = [(15.0, "flyable"), (50.0, "good"), (100.0, "epic")]

TIER_LABELS: List[str] = [label for _, label in TIERS]
