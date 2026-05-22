"""Built-in severity taxonomies for high-stakes domains.

A taxonomy bundles a four-level severity label set (negligible, minor,
major, critical) with a domain-specific cost vector calibrated against
the regulatory and economic evidence detailed in the paper appendix.

Use :func:`get_taxonomy` to retrieve a taxonomy by name and
:func:`severity_label_to_index` to map a severity label to its index
in the cost vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEVERITY_LABELS: list[str] = ["negligible", "minor", "major", "critical"]


@dataclass(frozen=True)
class Taxonomy:
    """Severity taxonomy: labels and dollar costs per level."""

    domain: str
    labels: tuple[str, ...]
    cost_levels: np.ndarray
    description: str = ""

    def index(self, label: str) -> int:
        """Return the position of ``label`` in :attr:`labels`."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise ValueError(
                f"Unknown severity label {label!r} for domain {self.domain!r}; "
                f"expected one of {self.labels}"
            ) from exc


_DEFAULT_LABELS = tuple(SEVERITY_LABELS)


# Cost vectors per the paper, Table 1 (severity-eval EMNLP draft).
# Calibration sources are documented in
# paper/severity-annotation-guidelines.tex.
_TAXONOMIES: dict[str, Taxonomy] = {
    "finance": Taxonomy(
        domain="finance",
        labels=_DEFAULT_LABELS,
        cost_levels=np.array([100.0, 1_000.0, 10_000.0, 100_000.0]),
        description="SEC SAB 99 materiality / SOX-906 calibration",
    ),
    "medical": Taxonomy(
        domain="medical",
        labels=_DEFAULT_LABELS,
        cost_levels=np.array([500.0, 5_000.0, 50_000.0, 500_000.0]),
        description="Clinical decision impact / malpractice settlement scale",
    ),
    "legal": Taxonomy(
        domain="legal",
        labels=_DEFAULT_LABELS,
        cost_levels=np.array([200.0, 2_000.0, 20_000.0, 200_000.0]),
        description="Due-diligence stratification / contractual exposure",
    ),
    "insurance": Taxonomy(
        domain="insurance",
        labels=_DEFAULT_LABELS,
        cost_levels=np.array([100.0, 2_000.0, 10_000.0, 250_000.0]),
        description="AMF complaints and reinsurance retention bands",
    ),
}


def list_domains() -> list[str]:
    """Return the names of the built-in taxonomies."""
    return sorted(_TAXONOMIES)


def get_taxonomy(domain: str) -> Taxonomy:
    """Look up a built-in taxonomy by domain name.

    Parameters
    ----------
    domain : str
        Domain identifier, case-insensitive. Accepts the canonical
        names plus a few common aliases (``law`` -> ``legal``,
        ``med`` -> ``medical``).
    """
    key = domain.strip().lower()
    aliases = {
        "law": "legal",
        "legal_nli": "legal",
        "legal_simplification": "legal",
        "med": "medical",
        "medical_qa": "medical",
        "healthcare": "medical",
        "finance_qa": "finance",
        "financial": "finance",
        "rag_insurance": "insurance",
    }
    key = aliases.get(key, key)
    if key not in _TAXONOMIES:
        raise KeyError(f"Unknown taxonomy {domain!r}. Available: {sorted(_TAXONOMIES)}")
    return _TAXONOMIES[key]


def register_taxonomy(taxonomy: Taxonomy, *, overwrite: bool = False) -> None:
    """Register a custom taxonomy for use with :func:`get_taxonomy`."""
    key = taxonomy.domain.strip().lower()
    if not overwrite and key in _TAXONOMIES:
        raise KeyError(f"Taxonomy {key!r} already registered; pass overwrite=True")
    _TAXONOMIES[key] = taxonomy


def severity_label_to_index(label: str) -> int:
    """Return the integer index of a severity label in the default ordering.

    Parameters
    ----------
    label : str
        One of ``negligible``, ``minor``, ``major``, ``critical``.
    """
    key = label.strip().lower()
    try:
        return SEVERITY_LABELS.index(key)
    except ValueError as exc:
        raise ValueError(
            f"Unknown severity label {label!r}; expected one of {SEVERITY_LABELS}"
        ) from exc
