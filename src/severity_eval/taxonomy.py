"""Severity taxonomies by domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DOMAINS = ["finance", "medical", "legal", "code_security", "moderation"]

SEVERITY_LABELS = ["negligible", "minor", "major", "critical"]


@dataclass
class DomainTaxonomy:
    """Severity taxonomy for a specific domain."""

    domain: str
    cost_levels: np.ndarray
    labels: list[str]
    description: str

    def validate_profile(self, pi: np.ndarray) -> bool:
        """Check that a severity profile is valid for this taxonomy."""
        pi = np.asarray(pi)
        return len(pi) == len(self.cost_levels) and np.isclose(pi.sum(), 1.0)


_TAXONOMIES: dict[str, DomainTaxonomy] = {
    "finance": DomainTaxonomy(
        domain="finance",
        cost_levels=np.array([100, 1000, 10000, 100000], dtype=np.float64),
        labels=SEVERITY_LABELS,
        description="Financial QA (SEC filings, earnings, ratios)",
    ),
    "medical": DomainTaxonomy(
        domain="medical",
        cost_levels=np.array([500, 5000, 50000, 500000], dtype=np.float64),
        labels=SEVERITY_LABELS,
        description="Medical calculations (dosage, triage, risk scores)",
    ),
    "legal": DomainTaxonomy(
        domain="legal",
        cost_levels=np.array([200, 2000, 20000, 200000], dtype=np.float64),
        labels=SEVERITY_LABELS,
        description="Legal contract review (clauses, liability, IP)",
    ),
    "code_security": DomainTaxonomy(
        domain="code_security",
        cost_levels=np.array([1000, 10000, 100000, 1000000], dtype=np.float64),
        labels=SEVERITY_LABELS,
        description="Code/Security (CVSS-based vulnerability costs)",
    ),
    "moderation": DomainTaxonomy(
        domain="moderation",
        cost_levels=np.array([50, 500, 5000, 50000], dtype=np.float64),
        labels=SEVERITY_LABELS,
        description="Content moderation (user safety, brand risk)",
    ),
}


def list_domains() -> list[str]:
    """Return all registered domain names."""
    return list(_TAXONOMIES.keys())


def get_taxonomy(domain: str) -> DomainTaxonomy:
    """Get the severity taxonomy for a domain.

    Parameters
    ----------
    domain : str
        Any registered domain name.

    Returns
    -------
    DomainTaxonomy
    """
    if domain not in _TAXONOMIES:
        available = list(_TAXONOMIES.keys())
        raise ValueError(f"Unknown domain '{domain}'. Available: {available}")
    return _TAXONOMIES[domain]


def register_taxonomy(
    domain: str,
    cost_levels: list[int] | list[float],
    labels: list[str] | None = None,
    description: str = "",
) -> DomainTaxonomy:
    """Register a custom domain taxonomy.

    Parameters
    ----------
    domain : str
        Domain name (e.g. 'insurance', 'autonomous_driving').
    cost_levels : list
        Dollar cost for each severity level.
    labels : list of str or None
        Names for each level. Auto-generated if None.
    description : str
        Human-readable description.

    Returns
    -------
    DomainTaxonomy
    """
    cost_arr = np.array(cost_levels, dtype=np.float64)
    K = len(cost_arr)
    if labels is None:
        if K == 4:
            labels = list(SEVERITY_LABELS)
        else:
            labels = [f"level_{i + 1}" for i in range(K)]
    elif len(labels) != K:
        raise ValueError(f"len(labels)={len(labels)} != len(cost_levels)={K}")

    tax = DomainTaxonomy(
        domain=domain,
        cost_levels=cost_arr,
        labels=labels,
        description=description,
    )
    _TAXONOMIES[domain] = tax
    return tax


def severity_label_to_index(label: str) -> int:
    """Convert a severity label to its index."""
    try:
        return SEVERITY_LABELS.index(label)
    except ValueError as err:
        raise ValueError(f"Unknown severity label '{label}'. Available: {SEVERITY_LABELS}") from err
