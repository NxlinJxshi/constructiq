# ============================================================
# MODULE: seed_mongodb.py
# PURPOSE: Populates the baselines collection with known-good historical
#          averages for each construction cost code so the detection
#          modules have reference data to compare against.
# PIPELINE STAGE: Database (setup / one-time seed)
# INPUTS: Hard-coded baseline constants derived from commercial solar
#         construction industry norms
# OUTPUTS: 5 documents written to the MongoDB 'baselines' collection
# ============================================================

"""Seeds MongoDB with synthetic timesheet records and baseline metrics."""

import sys
import os

# Add the project root to Python's module search path so that
# `from database.mongo_client import ...` works when this script is run
# directly from the scripts/ subdirectory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import only the baselines collection — we don't need the others for seeding.
from database.mongo_client import baselines

# ---------------------------------------------------------------------------
# BASELINE_DOCS — the reference dataset for anomaly detection
#
# Each document represents one cost code (a billing category in HCSS HeavyJob).
# The detection modules will later query these documents and flag any timesheet
# row whose values deviate significantly from these norms.
#
# Field explanations:
#   cost_code            — HCSS cost code string; used as the join key between
#                          timesheets and baselines
#   description          — Human-readable label for the cost code
#   avg_hours_per_worker — Expected productive hours per worker per shift.
#                          10.5 hrs reflects this project's 6:30 AM–5:30 PM
#                          schedule (11 hrs on-site minus 30-min unpaid lunch).
#   std_dev              — One standard deviation around the average.
#                          1.0 hr reflects the consistent daily schedule on this
#                          project; a row is flagged as an outlier if it falls
#                          more than 2 std_devs from the mean (~95% of normal data).
#   avg_equipment_hours  — Expected machine hours per shift for this activity.
#                          0.0 means no equipment is typically needed.
#   typical_labor_classes — HCSS labor class codes for workers normally billed
#                           to this cost code. Workers billed under an unexpected
#                           class are a potential billing error.
#                           SLSDL-FM  = Solar Foreman
#                           SLLS-L1   = Solar Laborer Level 1
#                           SLLS-L2   = Solar Laborer Level 2
#                           SLLS-A1   = Solar Apprentice Level 1
#                           SLLS-04   = Solar Laborer Group 4
#   avg_straight_time_hours — Average regular (non-overtime) hours per worker
#                             per shift. 8.0 hrs is the standard straight-time
#                             day for all cost codes on this project.
#   avg_overtime_hours      — Average overtime hours per worker per shift.
#                             2.5 hrs reflects the real data for labor codes
#                             (6:30 AM–5:30 PM minus 8 ST hrs = 2.5 OT hrs).
#                             Foreman admin runs only 0.5 OT as non-production.
#   common_equipment        — Equipment types typically deployed on this cost code,
#                             using the exact asset names from HCSS. An empty list
#                             means no equipment is expected; any equipment billed
#                             against such a code is an automatic anomaly.
#   equipment_operator_pairs — List of dicts pairing each equipment type with the
#                              labor class that typically operates it on this project,
#                              as observed in the HCSS sheets (equipment rows were
#                              nested under specific worker classifications).
#                              Used by the detection layer to flag equipment billed
#                              under an unexpected operator class.
#   units_completed         — Tracks production output for cost codes that report
#                             units. Fields:
#                               unit_type: "EA" (each) or "DA" (day-rate, no units)
#                               typical_units_per_shift: expected output per shift;
#                               0 means units were blank on the real HCSS sheet.
# ---------------------------------------------------------------------------
BASELINE_DOCS = [
    {
        # Foundation work involves forming, pouring, and stripping concrete.
        # Bobcat S510 is used for material movement; Chevy Colorado is the crew truck.
        "cost_code": "51214.4",
        "description": "Form, Pour, Strip Foundation",
        "avg_hours_per_worker": 10.5,       # 6:30 AM–5:30 PM shift on this project
        "std_dev": 1.0,                     # consistent daily schedule; low variance
        "avg_straight_time_hours": 8.0,     # standard 8-hr straight-time day
        "avg_overtime_hours": 2.5,          # remaining hrs after 8 ST = OT on this schedule
        "avg_equipment_hours": 4.5,         # Bobcat usage per shift for material staging
        "typical_labor_classes": ["SLSDL-FM", "SLLS-L1", "SLLS-L2", "SLLS-A1"],
        "common_equipment": ["BOBCAT S510", "CHEVY COLORADO"],
        # SLLS-A1 typically operates the Bobcat; SLLS-L1 drives the crew truck
        "equipment_operator_pairs": [
            {"equipment_type": "BOBCAT S510",    "typical_operator_class": "SLLS-A1"},
            {"equipment_type": "CHEVY COLORADO", "typical_operator_class": "SLLS-L1"},
        ],
        # Units were blank on real HCSS sheet for this code — day-rate billing
        "units_completed": {"unit_type": "DA", "typical_units_per_shift": 0},
    },
    {
        # Steel erection and panelizing is the most equipment-intensive task.
        # SkyTrak 10054 is a telehandler for lifting; JLG 400S is a boom lift
        # used to position workers at height during panel installation.
        "cost_code": "51234.5",
        "description": "Erect Steel and Panelize",
        "avg_hours_per_worker": 10.5,       # 6:30 AM–5:30 PM shift on this project
        "std_dev": 1.0,                     # consistent daily schedule; low variance
        "avg_straight_time_hours": 8.0,     # standard 8-hr straight-time day
        "avg_overtime_hours": 2.5,          # remaining hrs after 8 ST = OT on this schedule
        "avg_equipment_hours": 6.0,         # telehandler + boom lift hours per shift
        "typical_labor_classes": ["SLSDL-FM", "SLLS-L1", "SLLS-L2", "SLLS-04"],
        "common_equipment": ["SKYTRAK 10054", "JLG 400S BOOM LIFT"],
        # SLLS-A1 operates the SkyTrak (observed in HCSS sheet); SLLS-L2 on boom lift
        "equipment_operator_pairs": [
            {"equipment_type": "SKYTRAK 10054",      "typical_operator_class": "SLLS-A1"},
            {"equipment_type": "JLG 400S BOOM LIFT", "typical_operator_class": "SLLS-L2"},
        ],
        # 30 EA per shift sourced directly from the real HCSS timesheet for this code
        "units_completed": {"unit_type": "EA", "typical_units_per_shift": 30},
    },
    {
        # Yard unloading uses the SkyTrak telehandler to offload steel deliveries.
        # No boom lift needed since work stays at ground level.
        "cost_code": "51055.6",
        "description": "Yard Unload Steel",
        "avg_hours_per_worker": 10.5,       # 6:30 AM–5:30 PM shift on this project
        "std_dev": 1.0,                     # consistent daily schedule; low variance
        "avg_straight_time_hours": 8.0,     # standard 8-hr straight-time day
        "avg_overtime_hours": 2.5,          # remaining hrs after 8 ST = OT on this schedule
        "avg_equipment_hours": 3.5,         # telehandler hours for offloading and staging
        "typical_labor_classes": ["SLLS-L1", "SLLS-L2", "SLLS-04"],
        "common_equipment": ["SKYTRAK 10054"],
        # SLLS-A1 operates the SkyTrak for unloading (consistent with erection code)
        "equipment_operator_pairs": [
            {"equipment_type": "SKYTRAK 10054", "typical_operator_class": "SLLS-A1"},
        ],
        # Units were blank on real HCSS sheet for this code — day-rate billing
        "units_completed": {"unit_type": "DA", "typical_units_per_shift": 0},
    },
    {
        # Foreman administrative time covers planning, safety meetings, and
        # coordination. No field equipment is used on this code; any equipment
        # billed here is an automatic anomaly.
        "cost_code": "73020.7000",
        "description": "Foreman Administrative",
        "avg_hours_per_worker": 10.5,       # 6:30 AM–5:30 PM shift on this project
        "std_dev": 1.0,                     # consistent daily schedule; low variance
        "avg_straight_time_hours": 8.0,     # standard 8-hr straight-time day
        "avg_overtime_hours": 0.5,          # foreman admin runs minimal OT vs. field codes
        "avg_equipment_hours": 0.0,         # no equipment used for administrative tasks
        "typical_labor_classes": ["SLSDL-FM"],
        "common_equipment": [],             # no equipment expected on admin cost code
        "equipment_operator_pairs": [],     # no equipment means no operator pairings
        # Units were blank on real HCSS sheet for this code — day-rate billing
        "units_completed": {"unit_type": "DA", "typical_units_per_shift": 0},
    },
    {
        # Electrical rough-in involves conduit runs and wire pulling.
        # No heavy equipment needed; any equipment billed here is an anomaly.
        "cost_code": "51300.1",
        "description": "Electrical Rough-In",
        "avg_hours_per_worker": 10.5,       # 6:30 AM–5:30 PM shift on this project
        "std_dev": 1.0,                     # consistent daily schedule; low variance
        "avg_straight_time_hours": 8.0,     # standard 8-hr straight-time day
        "avg_overtime_hours": 2.5,          # remaining hrs after 8 ST = OT on this schedule
        "avg_equipment_hours": 1.5,         # hand-carried wire puller / conduit bender only
        "typical_labor_classes": ["SLSDL-FM", "SLLS-L1", "SLLS-A1"],
        "common_equipment": [],             # no heavy equipment expected on electrical code
        "equipment_operator_pairs": [],     # no equipment means no operator pairings
        # Units were blank on real HCSS sheet for this code — day-rate billing
        "units_completed": {"unit_type": "DA", "typical_units_per_shift": 0},
    },
]


def seed():
    """Insert all baseline documents into the MongoDB baselines collection.

    Uses insert_many for a single round-trip rather than looping over
    individual inserts. The returned result holds the list of generated _id
    values; we use its length as a cheap confirmation count.
    """
    # insert_many sends all documents in one batch — faster than 5 separate inserts.
    result = baselines.insert_many(BASELINE_DOCS)

    # len(inserted_ids) tells us exactly how many documents MongoDB accepted.
    # If any failed silently we'd see a count mismatch here.
    print(f"Inserted {len(result.inserted_ids)} baseline documents into 'baselines' collection.")


if __name__ == "__main__":
    # Run `python scripts/seed_mongodb.py` once during environment setup.
    # Re-running will create duplicate documents; clear the collection first
    # if you need to re-seed.
    seed()
