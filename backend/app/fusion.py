from backend.app.config import SAFE_THRESHOLD, PHISHING_THRESHOLD


def fuse_stage1_stage2(xgb_conf: float, content_conf: float) -> dict:
    score = max(xgb_conf, content_conf) * 0.6 + min(xgb_conf, content_conf) * 0.4
    return {"score": round(score, 4), "source": "url+content"}


def fuse_with_llm(base_score: float, llm_conf: float) -> dict:
    score = base_score * 0.4 + llm_conf * 0.6
    return {"score": round(score, 4), "source": "url+content+llm"}


def decide_tier(score: float) -> str:
    if score < SAFE_THRESHOLD:
        return "safe"
    if score > PHISHING_THRESHOLD:
        return "phishing"
    return "unsure"
