from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModeProfile:
    """Mode-specific knobs for grounded answering.

    Keep this intentionally small: it should only influence *formatting/synthesis policy*
    and safe generation parameters, not retrieval correctness.
    """

    mode: str

    # Whether the grounded agent may synthesize comparisons/differences when sources
    # contain explicit statements for each side.
    allow_comparative_synthesis: bool = True

    # Optional generation overrides. When None, the env defaults are used.
    grounded_model: Optional[str] = None
    grounded_temperature: Optional[float] = None


_DEFAULT_PROFILE = ModeProfile(mode="default")


_MODE_REGISTRY: dict[str, ModeProfile] = {
    # General chat: allow helpful synthesis, keep temperature low.
    "tourist_chat": ModeProfile(
        mode="tourist_chat",
        allow_comparative_synthesis=True,
        grounded_temperature=0.2,
    ),
    # FAQ: prefer determinism and low variability.
    "faq_first": ModeProfile(
        mode="faq_first",
        allow_comparative_synthesis=True,
        grounded_temperature=0.0,
    ),
    # Events/directory/coupons: factual + concise; avoid creative drift.
    "events": ModeProfile(
        mode="events",
        allow_comparative_synthesis=False,
        grounded_temperature=0.0,
    ),
    "directory": ModeProfile(
        mode="directory",
        allow_comparative_synthesis=False,
        grounded_temperature=0.0,
    ),
    "coupons": ModeProfile(
        mode="coupons",
        allow_comparative_synthesis=False,
        grounded_temperature=0.0,
    ),
}


def get_mode_profile(mode: str | None) -> ModeProfile:
    if not mode:
        return _DEFAULT_PROFILE
    return _MODE_REGISTRY.get(str(mode), _DEFAULT_PROFILE)
