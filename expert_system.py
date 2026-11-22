"""
expert_system.py
================
Rule-based Expert System core for football match & player performance analysis.

Components (classic expert-system architecture)
-----------------------------------------------
1. Working Memory       -> dict of facts (raw inputs, abstracted facts, inferred facts)
2. Knowledge Base       -> RULES: 10 production rules (IF-THEN)
3. Inference Engine     -> InferenceEngine.run(): forward chaining + conflict resolution
4. Explanation Facility -> FiredRule.evidence / InferenceResult.not_fired / .trace

Data-engineering helpers (CSV schema, validation, sample data, timeline analysis)
are also here. This module has NO Streamlit dependency, so it can be unit-tested
or run from the command line:  python expert_system.py
"""

from __future__ import annotations

import io
import operator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# 0. DATA SCHEMA (single source of truth for sliders, CSV columns and validation)
# ---------------------------------------------------------------------------

POSITIONS = ["Center Back", "Defensive Mid", "Playmaker", "Winger", "Striker"]

# Numeric facts: label, valid range, default and step. Float defaults => float facts.
FACT_SPECS: dict[str, dict[str, Any]] = {
    "Player_Sprint_Fatigue":       dict(label="Sprint fatigue (%)",                     min=0,   max=100,  default=60,   step=1),
    "Pass_Accuracy_Under_Press":   dict(label="Pass accuracy under press (%)",          min=0,   max=100,  default=78,   step=1),
    "Distance_Covered_km":         dict(label="Distance covered (km)",                  min=0.0, max=14.0, default=8.5,  step=0.1),
    "Ball_Loss_Count":             dict(label="Ball losses in danger zones (per 15')",  min=0,   max=15,   default=3,    step=1),
    "Opponent_Build_Up_Time_s":    dict(label="Opponent build-up time (s)",             min=3.0, max=25.0, default=12.0, step=0.5),
    "Our_PPDA":                    dict(label="Our PPDA (lower = more pressing)",       min=4.0, max=20.0, default=9.0,  step=0.5),
    "Space_Between_Lines":         dict(label="Space between lines (m)",                min=5.0, max=30.0, default=11.0, step=0.5),
    "Opponent_Through_Ball_Count": dict(label="Opponent through balls (per 15')",       min=0,   max=8,    default=1,    step=1),
    "Opponent_xG_Created":         dict(label="Opponent xG created (last 15')",         min=0.0, max=2.0,  default=0.15, step=0.05),
    "Midfield_Line_Height_m":      dict(label="Midfield line height (m from own goal)", min=20.0, max=70.0, default=45.0, step=1.0),
    "Substitutes_Available":       dict(label="Substitutions available",                min=0,   max=5,    default=3,    step=1),
}

# CSV layout: "Minute" is the time index (needed for the timeline chart); the rest are facts.
FACT_COLUMNS = ["Player_Name", "Player_Position", *FACT_SPECS.keys()]
CSV_COLUMNS = ["Minute", *FACT_COLUMNS]

DEFAULT_FACTS: dict[str, Any] = {
    "Player_Name": "A. Moreno",
    "Player_Position": "Playmaker",
    **{k: v["default"] for k, v in FACT_SPECS.items()},
}


def cast_fact(name: str, value: Any) -> Any:
    """Coerce a value to the type the schema expects (int / float / str)."""
    spec = FACT_SPECS.get(name)
    if spec is None:
        return str(value)
    return float(value) if isinstance(spec["default"], float) else int(round(float(value)))


def row_to_facts(row: Any) -> dict[str, Any]:
    """Turn one CSV row (dict or Series) into a working-memory fact dictionary."""
    return {c: cast_fact(c, row[c]) for c in FACT_COLUMNS}


# ---------------------------------------------------------------------------
# 1. WORKING MEMORY HELPERS
# ---------------------------------------------------------------------------

OPERATORS = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}

DX_PREFIX = "Diagnosis::"


def dx(name: str) -> str:
    """Fact key asserted when a diagnosis is made. Diagnoses-as-facts enable rule chaining."""
    return f"{DX_PREFIX}{name}"


# Diagnosis names (constants avoid typos when rules refer to each other)
DX_PRESSING = "Ineffective Pressing Block"
DX_EXHAUSTION = "Player Exhaustion Risk"
DX_CENTRAL = "Defensive Vulnerability in Central Corridor"
DX_SUB_ALERT = "Substitution Alert"
DX_COLLAPSE = "Systemic Tactical Collapse Risk"
DX_PASSIVE = "Passive Deep Block"
DX_PRESS_RESIST = "Press-Resistance Failure"
DX_THREAT = "Imminent Goal Conceding Threat"
DX_INJURY = "High Soft-Tissue Injury Risk"
DX_DISJOINT = "Disjointed High Press (Exposed Backline)"

# Thresholds used to turn raw tracking numbers into symbolic facts ("abstraction")
FAST_BUILDUP_MAX_SEC = 8.0
MEDIUM_BUILDUP_MAX_SEC = 15.0
HIGH_PRESS_MAX_PPDA = 8.0
MEDIUM_PRESS_MAX_PPDA = 12.0
HIGH_TURNOVERS_MIN = 8
MEDIUM_TURNOVERS_MIN = 4
FREQUENT_THROUGH_BALLS_MIN = 4
OCCASIONAL_THROUGH_BALLS_MIN = 2
LOW_MIDFIELD_LINE_M = 35.0
HIGH_MIDFIELD_LINE_M = 50.0
XG_DANGER = 0.75        # opponent xG per 15 min considered dangerous
XG_MEDIUM = 0.40


def _fmt(value: Any) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """A single IF-clause: <fact> <operator> <value>."""
    fact: str
    op: str
    value: Any

    def evaluate(self, wm: dict[str, Any]) -> tuple[bool, Any]:
        if self.fact not in wm:
            return False, None
        actual = wm[self.fact]
        return bool(OPERATORS[self.op](actual, self.value)), actual

    def describe(self) -> str:
        if self.fact.startswith(DX_PREFIX):
            return f'Diagnosis "{self.fact[len(DX_PREFIX):]}" has been made'
        return f"{self.fact} {self.op} {_fmt(self.value)}"


@dataclass
class Rule:
    """Production rule: IF (all conditions) THEN (diagnosis + recommendation)."""
    rule_id: str
    name: str
    conditions: list[Condition]
    diagnosis: str
    recommendation: str          # may contain {Fact_Name} placeholders
    severity: str                # "Critical" | "High" | "Medium" | "Info"
    rationale: str               # football reasoning behind the rule
    priority: int = 0            # conflict resolution: higher fires first
    source: str = "Core"         # "Core" or "Extended"

    def matches(self, wm: dict[str, Any]) -> bool:
        return all(c.evaluate(wm)[0] for c in self.conditions)

    def as_text(self) -> str:
        ifs = " AND ".join(c.describe() for c in self.conditions)
        return f'IF {ifs} THEN Diagnosis = "{self.diagnosis}" AND Recommendation = "{self.recommendation}"'


RULES: list[Rule] = [
    # ---- CORE RULES (R1-R3) -----------------------------------------------
    Rule(
        "R1", "Ineffective pressing vs fast build-up",
        [Condition("Opponent_Build_Up_Speed", "==", "Fast"), Condition("Our_Pressing_Intensity", "==", "Low")],
        DX_PRESSING, "Push defensive line higher and press holding midfielder.", "High",
        "A fast-building opponent is only stoppable if the first phase is disrupted. With low pressing "
        "intensity the opponent circulates the ball unchallenged and reaches the final third quickly. "
        "Pushing the line up and pressing the holding midfielder cuts the link between defence and attack.",
        priority=30,
    ),
    Rule(
        "R2", "Sprint fatigue + turnovers = exhaustion",
        [Condition("Player_Sprint_Fatigue", ">", 85), Condition("Ball_Loss_Turnovers", "==", "High")],
        DX_EXHAUSTION, "Immediate substitution required ({Player_Name}).", "Critical",
        "Sprint fatigue above 85 % means repeat-sprint capacity is nearly depleted. Combined with many ball "
        "losses, the drop in decision-making and technique is a signature of physical exhaustion.",
        priority=40,
    ),
    Rule(
        "R3", "Open central corridor",
        [Condition("Space_Between_Lines", ">", 15), Condition("Opponent_Through_Balls", "==", "Frequent")],
        DX_CENTRAL, "Compact the block by dropping the defensive midfielder deeper.", "High",
        "More than 15 m between defensive and midfield lines gives opposition playmakers room to receive "
        "between the lines. Frequent through balls show they are exploiting it. A deeper defensive "
        "midfielder shrinks the gap and shields the central corridor.",
        priority=30,
    ),
    # ---- EXTENDED RULES (R4-R6) ---------------------------------------------
    Rule(
        "R4", "Actionable substitution alert (chained from R2)",
        [Condition(dx(DX_EXHAUSTION), "==", True), Condition("Substitutes_Available", ">", 0)],
        DX_SUB_ALERT,
        "Prepare the replacement for {Player_Name} now; {Substitutes_Available} substitution(s) still available.",
        "Critical",
        "An exhaustion diagnosis only becomes actionable if the bench has substitutions left. "
        "This rule chains on the conclusion of R2.",
        priority=35, source="Extended",
    ),
    Rule(
        "R5", "Two defensive problems at once (chained from R1 + R3)",
        [Condition(dx(DX_PRESSING), "==", True), Condition(dx(DX_CENTRAL), "==", True)],
        DX_COLLAPSE,
        "Change shape (e.g. compact 4-5-1 mid-block) and relay new instructions at the next stoppage "
        "instead of making isolated fixes.",
        "Critical",
        "A broken press (R1) plus an open central corridor (R3) means the team is stretched vertically and "
        "cannot recover. Two separate adjustments would conflict, so a structural change is advised.",
        priority=50, source="Extended",
    ),
    Rule(
        "R6", "Passive deep block",
        [Condition("Midfield_Line_Height", "==", "Low"), Condition("Our_Pressing_Intensity", "==", "Low")],
        DX_PASSIVE, "Raise the midfield line by 8-10 m and trigger the press on backward passes.", "Medium",
        "A low midfield line plus low pressing invites sustained opponent possession near our box and "
        "gives away territory.",
        priority=10, source="Extended",
    ),
    # ---- NEW RULES (R7-R10) -------------------------------------------------
    Rule(
        "R7", "Press-resistance failure",
        [Condition("Pass_Accuracy_Under_Press", "<", 70), Condition("Ball_Loss_Turnovers", "==", "High")],
        DX_PRESS_RESIST,
        "Drop a deep-lying playmaker into the first line to create a passing triangle and give "
        "{Player_Name} a safe outlet.",
        "High",
        "Passing accuracy below 70 % under pressure plus a high number of dangerous-zone losses shows the "
        "team cannot play out of the press. A deep-lying playmaker adds a third passing angle.",
        priority=25, source="Extended",
    ),
    Rule(
        "R8", "Imminent goal threat (chained from R3)",
        [Condition("Opponent_xG_Created", ">", XG_DANGER), Condition(dx(DX_CENTRAL), "==", True)],
        DX_THREAT,
        "Give an immediate tactical-foul instruction to break up transitions and switch to a double pivot.",
        "Critical",
        "Opponent xG above 0.75 in 15 minutes means high-quality chances are already being created. Combined "
        "with the open central corridor (R3) the next goal is likely. Fouling stops the transition; a double "
        "pivot closes the corridor.",
        priority=60, source="Extended",
    ),
    Rule(
        "R9", "Workload overload / soft-tissue risk",
        [Condition("Distance_Covered_km", ">", 11.5), Condition("Player_Sprint_Fatigue", ">", 80)],
        DX_INJURY,
        "Apply workload management: reduce {Player_Name}'s high-intensity running or sub off this key "
        "starter ({Player_Position}).",
        "High",
        "Covering more than 11.5 km at over 80 % sprint fatigue is the profile where hamstring and "
        "calf strains rise sharply, because neuromuscular control degrades with fatigue.",
        priority=38, source="Extended",
    ),
    Rule(
        "R10", "Disjointed high press",
        [Condition("Our_Pressing_Intensity", "==", "High"), Condition("Space_Between_Lines", ">", 16)],
        DX_DISJOINT,
        "Either step the defensive line up by 7 m to close the gap, or drop into a mid-block "
        "to restore compactness.",
        "High",
        "Pressing high while the lines are more than 16 m apart means the front players press but the "
        "backline does not follow, leaving space behind the midfield for a single pass to exploit.",
        priority=28, source="Extended",
    ),
]


# ---------------------------------------------------------------------------
# 3. INFERENCE ENGINE (forward chaining)
# ---------------------------------------------------------------------------

@dataclass
class FiredRule:
    """Record of one rule firing, kept for the Explanation Facility."""
    rule: Rule
    cycle: int
    evidence: list[tuple[Condition, Any]]
    recommendation: str


@dataclass
class InferenceResult:
    working_memory: dict[str, Any]
    origin: dict[str, str]
    fired: list[FiredRule]
    not_fired: list[tuple[Rule, list[str]]]
    trace: list[str] = field(default_factory=list)


class InferenceEngine:
    """
    Forward-chaining (data-driven) inference engine.

    1. ABSTRACT numeric measurements into symbolic facts (e.g. PPDA 14 -> "Low").
    2. Repeat: MATCH applicable rules -> RESOLVE conflicts (highest priority, then KB order)
       -> FIRE one rule (assert its diagnosis as a fact, possibly enabling other rules)
       until the agenda is empty. Each rule fires at most once (refraction).
    """

    def __init__(self, rules: list[Rule]):
        self.rules = rules

    @staticmethod
    def _abstract(wm: dict, origin: dict, trace: list[str]) -> None:
        def put(name: str, value: Any, why: str) -> None:
            wm[name] = value
            origin[name] = "Abstracted"
            trace.append(f"[ABSTRACT] {name} = {_fmt(value)}   ({why})")

        t = wm["Opponent_Build_Up_Time_s"]
        put("Opponent_Build_Up_Speed",
            "Fast" if t <= FAST_BUILDUP_MAX_SEC else "Medium" if t <= MEDIUM_BUILDUP_MAX_SEC else "Slow",
            f"{t:.1f} s to reach final third")
        p = wm["Our_PPDA"]
        put("Our_Pressing_Intensity",
            "High" if p <= HIGH_PRESS_MAX_PPDA else "Medium" if p <= MEDIUM_PRESS_MAX_PPDA else "Low",
            f"PPDA {p:.1f}; lower PPDA = more intense")
        n = wm["Ball_Loss_Count"]
        put("Ball_Loss_Turnovers",
            "High" if n >= HIGH_TURNOVERS_MIN else "Medium" if n >= MEDIUM_TURNOVERS_MIN else "Low",
            f"{n} dangerous-zone losses / 15 min")
        tb = wm["Opponent_Through_Ball_Count"]
        put("Opponent_Through_Balls",
            "Frequent" if tb >= FREQUENT_THROUGH_BALLS_MIN else "Occasional" if tb >= OCCASIONAL_THROUGH_BALLS_MIN else "Rare",
            f"{tb} through balls / 15 min")
        h = wm["Midfield_Line_Height_m"]
        put("Midfield_Line_Height",
            "Low" if h < LOW_MIDFIELD_LINE_M else "High" if h > HIGH_MIDFIELD_LINE_M else "Medium",
            f"{h:.0f} m from own goal")
        x = wm["Opponent_xG_Created"]
        put("Opponent_Threat_Level",
            "High" if x > XG_DANGER else "Medium" if x > XG_MEDIUM else "Low",
            f"opponent xG {x:.2f} in last 15 min")

    def run(self, raw_facts: dict[str, Any]) -> InferenceResult:
        wm = dict(raw_facts)
        origin = {k: "Input" for k in wm}
        trace: list[str] = []
        fired: list[FiredRule] = []
        fired_ids: set[str] = set()

        self._abstract(wm, origin, trace)

        cycle = 0
        while True:
            cycle += 1
            agenda = [r for r in self.rules if r.rule_id not in fired_ids and r.matches(wm)]
            if not agenda:
                trace.append(f"[CYCLE {cycle}] Agenda empty -> no more rules applicable. Inference complete.")
                break
            agenda.sort(key=lambda r: -r.priority)   # stable: ties keep KB order
            rule = agenda[0]
            trace.append(f"[CYCLE {cycle}] Agenda: {[r.rule_id for r in agenda]} "
                         f"-> selected {rule.rule_id} (highest priority {rule.priority})")

            evidence = [(c, c.evaluate(wm)[1]) for c in rule.conditions]
            wm[dx(rule.diagnosis)] = True
            origin[dx(rule.diagnosis)] = f"Inferred by {rule.rule_id}"
            fired.append(FiredRule(rule, cycle, evidence, rule.recommendation.format_map(wm)))
            fired_ids.add(rule.rule_id)
            trace.append(f'[FIRE] {rule.rule_id} "{rule.name}" -> asserted fact {dx(rule.diagnosis)} = True')

        not_fired: list[tuple[Rule, list[str]]] = []
        for r in self.rules:
            if r.rule_id in fired_ids:
                continue
            reasons = []
            for c in r.conditions:
                ok, actual = c.evaluate(wm)
                if not ok:
                    shown = "not established" if actual is None else _fmt(actual)
                    reasons.append(f"{c.describe()}  (actual: {shown})")
            not_fired.append((r, reasons))
        return InferenceResult(wm, origin, fired, not_fired, trace)


# ---------------------------------------------------------------------------
# 4. DATA ENGINEERING: sample data, CSV validation, timeline analysis
# ---------------------------------------------------------------------------

def sample_dataframe() -> pd.DataFrame:
    """A 90-minute story for one player (15-minute windows) that exercises most rules."""
    cols = ["Minute", "Player_Sprint_Fatigue", "Pass_Accuracy_Under_Press", "Distance_Covered_km",
            "Ball_Loss_Count", "Opponent_Build_Up_Time_s", "Our_PPDA", "Space_Between_Lines",
            "Opponent_Through_Ball_Count", "Opponent_xG_Created", "Midfield_Line_Height_m",
            "Substitutes_Available"]
    data = [
        (15, 22, 84, 1.9,  2, 14.0,  7.5, 11.0, 0, 0.05, 55.0, 5),
        (30, 38, 80, 3.8,  3, 13.0,  7.0, 13.0, 1, 0.12, 53.0, 5),
        (45, 52, 76, 5.7,  4, 11.0,  7.8, 16.5, 2, 0.20, 51.0, 5),   # R10: press high, lines split
        (60, 68, 71, 7.7,  6,  7.5, 12.5, 15.5, 3, 0.38, 44.0, 4),   # R1: fast build-up vs weak press
        (75, 84, 64, 9.8,  9,  7.0, 14.0, 17.0, 5, 0.82, 34.0, 3),   # R1, R3, R5, R6, R7, R8
        (90, 92, 58, 11.9, 11, 6.5, 15.0, 18.5, 6, 1.10, 31.0, 2),   # almost every rule
    ]
    df = pd.DataFrame(data, columns=cols)
    df.insert(1, "Player_Name", "A. Moreno")
    df.insert(2, "Player_Position", "Defensive Mid")
    return df[CSV_COLUMNS]


def sample_csv_bytes() -> bytes:
    return sample_dataframe().to_csv(index=False).encode("utf-8")


def validate_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """Validate and clean an uploaded CSV. Returns (clean_df | None, errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FACT_COLUMNS if c not in df.columns]
    if missing:
        return None, [f"Missing required column(s): {', '.join(missing)}"], warnings

    if "Minute" not in df.columns:
        df.insert(0, "Minute", [15 * (i + 1) for i in range(len(df))])
        warnings.append("No 'Minute' column found: assumed 15-minute intervals.")

    numeric_cols = ["Minute", *FACT_SPECS.keys()]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_rows = df[numeric_cols].isna().any(axis=1)
    if bad_rows.any():
        warnings.append(f"Dropped {int(bad_rows.sum())} row(s) with missing or non-numeric values.")
        df = df[~bad_rows]
    if df.empty:
        return None, ["The file contains no valid rows."], warnings

    for col, spec in FACT_SPECS.items():
        outside = int(((df[col] < spec["min"]) | (df[col] > spec["max"])).sum())
        if outside:
            warnings.append(f"{outside} value(s) in '{col}' were outside [{spec['min']}, {spec['max']}] and were clipped.")
            df[col] = df[col].clip(spec["min"], spec["max"])

    df["Player_Name"] = df["Player_Name"].astype(str).str.strip()
    df["Player_Position"] = df["Player_Position"].astype(str).str.strip()
    bad_pos = ~df["Player_Position"].isin(POSITIONS)
    if bad_pos.any():
        warnings.append(f"{int(bad_pos.sum())} unknown position(s) replaced by 'Playmaker'. Valid: {', '.join(POSITIONS)}.")
        df.loc[bad_pos, "Player_Position"] = "Playmaker"

    return df[CSV_COLUMNS].sort_values("Minute").reset_index(drop=True), errors, warnings


def analyze_timeline(df: pd.DataFrame, engine: InferenceEngine) -> list[InferenceResult]:
    """Run the expert system on every row (time window) of a validated DataFrame."""
    return [engine.run(row_to_facts(r)) for r in df.to_dict("records")]


# ---------------------------------------------------------------------------
# Command-line demo:  python expert_system.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = InferenceEngine(RULES)
    frame = sample_dataframe()
    for (_, row), res in zip(frame.iterrows(), analyze_timeline(frame, engine)):
        print(f"Minute {int(row['Minute']):>2}: {[f.rule.rule_id for f in res.fired] or 'no alerts'}")
