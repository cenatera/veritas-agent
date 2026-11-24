"""Pruebas de la estimación de coste."""

from src.costs import estimate_high_usd
from src.orchestrator import AuditLimits


def test_estimate_grows_with_every_limit() -> None:
    base = estimate_high_usd("x", AuditLimits(max_claims=1, max_source_chars=1000))
    assert 0 < base < estimate_high_usd("x" * 9000, AuditLimits(max_claims=1, max_source_chars=1000))
    assert base < estimate_high_usd("x", AuditLimits(max_claims=2, max_source_chars=1000))
    assert base < estimate_high_usd("x", AuditLimits(max_claims=1, max_source_chars=2000))


def test_estimate_covers_the_measured_live_run() -> None:
    # La prueba real costó 0,57 USD con estos límites y tres afirmaciones.
    assert estimate_high_usd("x" * 230, AuditLimits(max_claims=3, max_source_chars=150_000)) > 0.57
