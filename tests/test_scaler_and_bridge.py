import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.balancer import scale_substitution_text  # noqa: E402
from app.engine import FlavorEngine  # noqa: E402


def test_scale_substitution_fractions():
    raw = "3/4 cup plain yogurt + 1/4 cup water (rest 10 min)"
    doubled = scale_substitution_text(raw, multiplier=2.0)
    halved = scale_substitution_text(raw, multiplier=0.5)
    assert doubled == "1 1/2 cup plain yogurt + 1/2 cup water (rest 10 min)"
    assert halved == "3/8 cup plain yogurt + 1/8 cup water (rest 10 min)"


def test_flavor_bridge_connectivity():
    engine = FlavorEngine()
    bridges = engine.find_flavor_bridges(["tomato", "garlic", "paneer"], top_k=3)
    assert len(bridges) > 0
    assert bridges[0]["ingredient"] not in {"tomato", "garlic", "paneer"}
