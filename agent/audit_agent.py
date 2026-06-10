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

Payload cap: only the top NARRATE_LIMIT flags (highest severity first) are sent
to Gemini to keep request size and latency bounded.  All remaining flags are
returned with their original rule-generated explanations.  The full list is
always returned — the cap limits only what is sent to Gemini, never what comes back.

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

# Severity sort order — lower number = shown first.
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

# Maximum number of flags sent to Gemini in a single narration request.
# Sending hundreds of flags in one prompt causes multi-minute hangs; capping
# to the most severe flags keeps latency under a few seconds.
# Override at runtime with the GEMINI_NARRATE_LIMIT env var.
NARRATE_LIMIT = int(os.environ.get("GEMINI_NARRATE_LIMIT", "25"))


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


def _sort_by_severity(flags: list[dict]) -> list[dict]:
    """Return a new list sorted high → medium → low, preserving input order within each group."""
    return sorted(flags, key=lambda f: _SEVERITY_ORDER.get(f.get("severity", "low"), 2))


def _fallback(flags: list[dict], reason: str) -> dict:
    """Return original flags sorted by severity with a generic summary.

    Used whenever the Gemini call fails or its output cannot be validated.
    Logs the reason so the operator knows why narration was skipped.
    Total flags in == total flags out is preserved.
    """
    logger.warning("Gemini narration skipped — using rule-generated explanations. Reason: %s", reason)

    sorted_flags = _sort_by_severity(flags)

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
    """Send the top NARRATE_LIMIT flags to Gemini and return ALL flags severity-sorted.

    Only the highest-severity flags (up to NARRATE_LIMIT) are sent to Gemini for
    plain-English narration.  The rest are returned with their original rule-generated
    explanations unchanged.  The full input list is always returned — the cap only
    limits the Gemini payload, never the output.

    Returns {"flags": [...all flags, severity-sorted...], "summary": str}.

    On any failure (missing credentials, API error, JSON parse error, count
    mismatch on the narrated subset), falls back to returning the full input list
    sorted by severity with original explanations.
    """
    if not flags:
        return {"flags": [], "summary": "No anomalies were detected in this timesheet."}

    # Sort ALL flags high → medium → low.  The subset sent to Gemini is the top slice.
    sorted_flags = _sort_by_severity(flags)
    subset = sorted_flags[:NARRATE_LIMIT]

    # ── Initialize Vertex AI ──────────────────────────────────────────────────
    try:
        _init_vertexai()
    except Exception as exc:
        return _fallback(flags, f"Vertex AI init failed: {exc}")

    # ── Build prompt for the subset only ─────────────────────────────────────
    try:
        flags_json  = json.dumps(subset, indent=2, default=str)
        user_prompt = AUDIT_USER_TEMPLATE.format(
            flag_count=len(subset),
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

    enriched_subset = parsed.get("flags")
    summary         = parsed.get("summary", "")

    if not isinstance(enriched_subset, list):
        return _fallback(flags, "Gemini response missing 'flags' array")

    # ── Invariant: subset count must be preserved ─────────────────────────────
    # Gemini must return exactly as many flags as it was sent.
    if len(enriched_subset) != len(subset):
        return _fallback(
            flags,
            f"Subset count mismatch: sent {len(subset)}, received {len(enriched_subset)}"
        )

    # ── Ensure all required §5 keys are present in enriched output ────────────
    required = {"flag_id", "record_id", "flag_type", "severity", "entity_type",
                "field", "observed", "explanation"}
    for enriched in enriched_subset:
        missing_keys = required - set(enriched.keys())
        if missing_keys:
            return _fallback(
                flags,
                f"Enriched flag is missing keys {missing_keys}; falling back."
            )

    # ── Merge narrated explanations back into the full sorted list ────────────
    # Build a lookup of flag_id → Gemini-narrated flag for the subset.
    narrated_by_id = {f["flag_id"]: f for f in enriched_subset}

    merged: list[dict] = []
    for flag in sorted_flags:
        narrated = narrated_by_id.get(flag["flag_id"])
        if narrated is not None:
            # Replace only the explanation; all other fields come from the original.
            updated = dict(flag)
            updated["explanation"] = narrated["explanation"]
            merged.append(updated)
        else:
            merged.append(flag)

    return {"flags": merged, "summary": summary}
