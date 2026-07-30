import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.fusion import fuse_stage1_stage2, fuse_with_llm, decide_tier


def test_fuse_stage1_stage2_agree():
    result = fuse_stage1_stage2(0.9, 0.85)
    assert 0 < result["score"] < 1
    assert result["source"] == "url+content"


def test_fuse_stage1_stage2_disagree():
    result = fuse_stage1_stage2(0.9, 0.1)
    assert 0 < result["score"] < 1
    assert result["source"] == "url+content"


def test_fuse_stage1_stage2_both_low():
    result = fuse_stage1_stage2(0.2, 0.1)
    assert result["score"] < 0.3


def test_fuse_stage1_stage2_both_high():
    result = fuse_stage1_stage2(0.95, 0.9)
    assert result["score"] > 0.5


def test_fuse_with_llm():
    result = fuse_with_llm(0.8, 0.9)
    assert result["source"] == "url+content+llm"
    assert result["score"] == round(0.8 * 0.4 + 0.9 * 0.6, 4)


def test_fuse_with_llm_low_confidence():
    result = fuse_with_llm(0.1, 0.2)
    assert result["score"] < 0.3


def test_decide_tier_safe():
    assert decide_tier(0.1) == "safe"
    assert decide_tier(0.29) == "safe"


def test_decide_tier_unsure():
    assert decide_tier(0.5) == "unsure"
    assert decide_tier(0.31) == "unsure"
    assert decide_tier(0.69) == "unsure"


def test_decide_tier_phishing():
    assert decide_tier(0.8) == "phishing"
    assert decide_tier(0.71) == "phishing"


def test_decide_tier_boundary_safe():
    assert decide_tier(0.30) == "unsure"


def test_decide_tier_boundary_phishing():
    assert decide_tier(0.70) == "unsure"
