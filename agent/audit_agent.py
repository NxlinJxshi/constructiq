# ============================================================
# MODULE: agent/audit_agent.py
# PURPOSE: Call Gemini via Vertex AI to narrate and rank the flags
#          produced by the detection pipeline, then return enriched
#          flags and an audit summary.
# PIPELINE STAGE: Agent (step 3 — called by main.run_audit)
# INPUTS: List of flag dicts conforming to the §5 schema
# OUTPUTS: {"flags": [...enriched flags...], "summary": str}
# ============================================================

"""
Sends detection flags to Gemini for plain-English narration and severity ranking.

HARD BOUNDARY: Gemini narrates and ranks only.  Flag count in == flag count out.
It must not invent, remove, or alter flags.  If Gemini fails, returns the
original flags unchanged — the pipeline never loses flags because of a
model call.

Authentication uses the same GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_REGION /
GOOGLE_APPLICATION_CREDENTIALS already used by the Vertex IsolationForest
endpoint.  No separate Gemini API key is needed or permitted.
"""

from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv

from agent.prompts import AUDIT_SYSTEM_INSTRUCTION, AUDIT_USER_TEMPLATE

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini model to use.  Override with GEMINI_MODEL env var for testing.
_DEFAULT_MODEL = "gemini-2.5-flash"


def _init_vertexai() -> None:
    """Initialize the Vertex AI SDK with project and region from environment."""
    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    region  = os.environ.get("GOOGLE_CLOUD_REGION",  "").strip()
    if not project or not region:
        raise EnvironmentError(
            "GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_REGION must be set to use Gemini."
        )
    import vertexai
    vertexai.init(project=project, location=region)


def _strip_json_fences(text: str) -> str:
    """Remove leading/trailing ```json … ``` fences that Gemini sometimes adds."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _fallback(flags: list[dict], reason: str) -> dict:
    """Return original flags sorted by severity with a generic summary.

    Used whenever the Gemini call fails or its output cannot be validated.
    Logs the reason so the operator knows why narration was skipped.
    """
    logger.warning("Gemini narration skipped — using rule-generated explanations. Reason: %s", reason)

    order = {"high": 0, "medium": 1, "low": 2}
    sorted_flags = sorted(flags, key=lambda f: order.get(f.get("severity", "low"), 2))

    counts = {"high": 0, "medium": 0, "low": 0}
    for f in flags:
        sev = f.get("severity", "low")
        counts[sev] = counts.get(sev, 0) + 1

    summary = (
        f"Automated audit complete. Found {len(flags)} anomaly flag(s): "
        f"{counts.get('high', 0)} high, {counts.get('medium', 0)} medium, "
        f"{counts.get('low', 0)} low severity. "
        "AI narration was unavailable — see individual flag explanations for details."
    )
    return {"flags": sorted_flags, "summary": summary}


def narrate_and_rank(flags: list[dict]) -> dict:
    """Send flags to Gemini for plain-English narration and severity ranking.

    Returns {"flags": [...enriched flags...], "summary": str}.
    Each enriched flag retains all §5 keys and gains a rewritten 'explanation'.
    Flag count in == flag count out — this invariant is asserted before returning.

    On any failure (missing credentials, API error, JSON parse error, count
    mismatch), falls back to the original flags sorted by severity.
    """
    if not flags:
        return {"flags": [], "summary": "No anomalies were detected in this timesheet."}

    # ── Initialize Vertex AI ──────────────────────────────────────────────────
    try:
        _init_vertexai()
    except Exception as exc:
        return _fallback(flags, f"Vertex AI init failed: {exc}")

    # ── Build prompt ──────────────────────────────────────────────────────────
    try:
        flags_json = json.dumps(flags, indent=2, default=str)
        user_prompt = AUDIT_USER_TEMPLATE.format(
            flag_count=len(flags),
            flags_json=flags_json,
        )
    except Exception as exc:
        return _fallback(flags, f"Prompt construction failed: {exc}")

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        from vertexai.generative_models import GenerativeModel

        model_name = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip()
        model      = GenerativeModel(
            model_name,
            system_instruction=AUDIT_SYSTEM_INSTRUCTION,
        )
        response = model.generate_content(user_prompt)
        raw_text = response.text
    except Exception as exc:
        return _fallback(flags, f"Gemini API call failed: {exc}")

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        cleaned = _strip_json_fences(raw_text)
        parsed  = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        return _fallback(flags, f"JSON parse error ({exc}) — raw response: {raw_text[:200]!r}")

    enriched_flags = parsed.get("flags")
    summary        = parsed.get("summary", "")

    if not isinstance(enriched_flags, list):
        return _fallback(flags, "Gemini response missing 'flags' array")

    # ── Invariant: flag count must be preserved ───────────────────────────────
    if len(enriched_flags) != len(flags):
        return _fallback(
            flags,
            f"Flag count mismatch: sent {len(flags)}, received {len(enriched_flags)}"
        )

    # ── Ensure all required §5 keys are present in enriched output ────────────
    required = {"flag_id", "record_id", "flag_type", "severity", "entity_type",
                "field", "observed", "explanation"}
    for enriched in enriched_flags:
        missing_keys = required - set(enriched.keys())
        if missing_keys:
            return _fallback(
                flags,
                f"Enriched flag is missing keys {missing_keys}; falling back."
            )

    return {"flags": enriched_flags, "summary": summary}
