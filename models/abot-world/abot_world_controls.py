"""Preserve ABot-World's native held-key and short-tap semantics."""

from __future__ import annotations

KEY_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
"""Native action-channel order expected by ``CausalInferencePipeline.set_act``."""

KEY_CHOICES = list(KEY_ORDER)
"""Schema choices for the public key-state command."""

CONFLICT_GROUPS = (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L"))
"""Opposing movement and view controls from the upstream web client."""


def resolve_conflicts(
    keys: frozenset[str], high_priority: frozenset[str]
) -> frozenset[str]:
    """Keep a newly activated key when it conflicts with an older held key."""
    result = set(keys)
    for first, second in CONFLICT_GROUPS:
        if first not in result or second not in result:
            continue
        first_is_high = first in high_priority
        second_is_high = second in high_priority
        if first_is_high and not second_is_high:
            result.discard(second)
        elif second_is_high and not first_is_high:
            result.discard(first)
    return frozenset(result)


def update_key_state(
    pressed_keys: frozenset[str],
    activated_keys: frozenset[str],
    *,
    key: str,
    pressed: bool,
) -> tuple[frozenset[str], frozenset[str]]:
    """Apply one physical key transition without losing a short tap."""
    if pressed:
        pressed_keys = pressed_keys.union((key,))
        activated_keys = resolve_conflicts(
            activated_keys.union((key,)),
            high_priority=frozenset((key,)),
        )
    else:
        pressed_keys = pressed_keys.difference((key,))
    return pressed_keys, activated_keys


def sample_key_snapshot(
    pressed_keys: frozenset[str], activated_keys: frozenset[str]
) -> tuple[dict[str, bool], frozenset[str]]:
    """Sample one upstream action block and consume queued short taps."""
    sampled = resolve_conflicts(
        pressed_keys.union(activated_keys),
        high_priority=activated_keys,
    )
    return {key: key in sampled for key in KEY_ORDER}, sampled
