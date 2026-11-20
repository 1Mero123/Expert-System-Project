"""
expert_system.py
================
Rule-based Expert System core for football match & player performance analysis.

Components (classic expert-system architecture)
-----------------------------------------------
1. Working Memory      -> a dict of facts (raw inputs, abstracted facts, inferred facts)
2. Knowledge Base      -> RULES: a list of production rules (IF-THEN)
3. Inference Engine    -> InferenceEngine.run(): forward chaining with conflict resolution
4. Explanation Facility-> FiredRule.evidence / InferenceResult.not_fired / InferenceResult.trace

This module has NO dependency on Streamlit, so it can be unit-tested or run from
the command line:  python expert_system.py
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 1. WORKING MEMORY HELPERS
# ---------------------------------------------------------------------------

OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

DX_PREFIX = "Diagnosis::"


def dx(name: str) -> str:
    """Fact key that is asserted into working memory when a diagnosis is made.
    Letting diagnoses become facts is what makes rule *chaining* possible."""
    return f"{DX_PREFIX}{name}"


# Diagnosis names (constants avoid typos when rules refer to each other)
DX_PRESSING = "Ineffective Pressing Block"
DX_EXHAUSTION = "Player Exhaustion Risk"
DX_CENTRAL = "Defensive Vulnerability in Central Corridor"
DX_SUB_ALERT = "Substitution Alert"
DX_COLLAPSE = "Systemic Tactical Collapse Risk"
DX_PASSIVE = "Passive Deep Block"

# Thresholds used to turn raw tracking numbers into symbolic facts ("abstraction")
FAST_BUILDUP_MAX_SEC = 8.0      # opponent reaches final third in <= 8 s  -> "Fast"
MEDIUM_BUILDUP_MAX_SEC = 15.0   # <= 15 s -> "Medium", otherwise "Slow"
HIGH_PRESS_MAX_PPDA = 8.0       # PPDA <= 8  -> "High" pressing (lower PPDA = more intense)
MEDIUM_PRESS_MAX_PPDA = 12.0    # PPDA <= 12 -> "Medium", otherwise "Low"
HIGH_TURNOVERS_MIN = 8          # >= 8 dangerous-zone losses per 15 min -> "High"
MEDIUM_TURNOVERS_MIN = 4        # >= 4 -> "Medium", otherwise "Low"
FREQUENT_THROUGH_BALLS_MIN = 4  # >= 4 opponent through balls per 15 min -> "Frequent"
OCCASIONAL_THROUGH_BALLS_MIN = 2
LOW_MIDFIELD_LINE_M = 35.0      # midfield line < 35 m from own goal -> "Low"
HIGH_MIDFIELD_LINE_M = 50.0     # > 50 m -> "High", otherwise "Medium"


def _fmt(value: Any) -> str:
    """Pretty-print a value for explanations."""
    return f'"{value}"' if isinstance(value, str) else str(value)


# ---------------------------------------------------------------------------
# 2. KNOWLEDGE BASE: CONDITIONS, RULES
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """A single IF-clause: <fact> <operator> <value>."""
    fact: str
    op: str
    value: Any

    def evaluate(self, wm: dict[str, Any]) -> tuple[bool, Any]:
        """Return (satisfied?, actual value found in working memory)."""
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
    """A production rule: IF (all conditions) THEN (diagnosis + recommendation)."""
    rule_id: str
    name: str
    conditions: list[Condition]
    diagnosis: str
    recommendation: str          # may contain {Fact_Name} placeholders
    severity: str                # "Critical" | "High" | "Medium" | "Info"
    rationale: str               # the football reasoning ("why") behind the rule
    priority: int = 0            # conflict resolution: higher fires first
    source: str = "Core"         # "Core" (required spec) or "Extended"

    def matches(self, wm: dict[str, Any]) -> bool:
        return all(c.evaluate(wm)[0] for c in self.conditions)

    def as_text(self) -> str:
        ifs = " AND ".join(c.describe() for c in self.conditions)
        return f'IF {ifs} THEN Diagnosis = "{self.diagnosis}" AND Recommendation = "{self.recommendation}"'


# The Knowledge Base ---------------------------------------------------------
RULES: list[Rule] = [
    # ---- CORE RULES (from the project specification) ----------------------
    Rule(
        rule_id="R1",
        name="Ineffective pressing vs fast build-up",
        conditions=[
            Condition("Opponent_Build_Up_Speed", "==", "Fast"),
            Condition("Our_Pressing_Intensity", "==", "Low"),
        ],
        diagnosis=DX_PRESSING,
        recommendation="Push defensive line higher and press holding midfielder.",
        severity="High",
        rationale=(
            "A fast-building opponent is only stoppable if the first phase is "
            "disrupted. With low pressing intensity the opponent circulates the ball "
            "unchallenged and reaches the final third quickly. Pushing the line up and "
            "pressing the holding midfielder cuts the link between defence and attack."
        ),
        priority=30,
    ),
    Rule(
        rule_id="R2",
        name="Sprint fatigue + turnovers = exhaustion",
        conditions=[
            Condition("Player_Sprint_Fatigue", ">", 85),
            Condition("Ball_Loss_Turnovers", "==", "High"),
        ],
        diagnosis=DX_EXHAUSTION,
        recommendation="Immediate substitution required ({Player_Name}).",
        severity="Critical",
        rationale=(
            "Sprint fatigue above 85 % means the player's repeat-sprint capacity is "
            "nearly depleted. Combined with many ball losses, the drop in decision "
            "making and technical quality is a signature of physical exhaustion and "
            "a high injury / goal-conceding risk."
        ),
        priority=40,
    ),
    Rule(
        rule_id="R3",
        name="Open central corridor",
        conditions=[
            Condition("Space_Between_Lines", ">", 15),
            Condition("Opponent_Through_Balls", "==", "Frequent"),
        ],
        diagnosis=DX_CENTRAL,
        recommendation="Compact the block by dropping the defensive midfielder deeper.",
        severity="High",
        rationale=(
            "More than 15 m between the defensive and midfield lines gives the "
            "opponent's playmakers room to receive between the lines. Frequent "
            "through balls show they are already exploiting it. A deeper defensive "
            "midfielder shrinks that gap and shields the central corridor."
        ),
        priority=30,
    ),
    # ---- EXTENDED RULES (demonstrate chaining on inferred facts) ----------
    Rule(
        rule_id="R4",
        name="Actionable substitution alert (chained from R2)",
        conditions=[
            Condition(dx(DX_EXHAUSTION), "==", True),
            Condition("Substitutes_Available", ">", 0),
        ],
        diagnosis=DX_SUB_ALERT,
        recommendation=(
            "Prepare the replacement for {Player_Name} now; "
            "{Substitutes_Available} substitution(s) still available."
        ),
        severity="Critical",
        rationale=(
            "An exhaustion diagnosis only becomes actionable if the bench has "
            "substitutions left. This rule chains on the conclusion of R2."
        ),
        priority=35,
        source="Extended",
    ),
    Rule(
        rule_id="R5",
        name="Two defensive problems at once (chained from R1 + R3)",
        conditions=[
            Condition(dx(DX_PRESSING), "==", True),
            Condition(dx(DX_CENTRAL), "==", True),
        ],
        diagnosis=DX_COLLAPSE,
        recommendation=(
            "Change shape (e.g. compact 4-5-1 mid-block) and relay new instructions "
            "at the next stoppage instead of making isolated fixes."
        ),
        severity="Critical",
        rationale=(
            "A broken press (R1) together with an open central corridor (R3) means "
            "the team is stretched vertically and cannot recover. Two separate "
            "adjustments would conflict, so a structural change is advised."
        ),
        priority=50,
        source="Extended",
    ),
    Rule(
        rule_id="R6",
        name="Passive deep block",
        conditions=[
            Condition("Midfield_Line_Height", "==", "Low"),
            Condition("Our_Pressing_Intensity", "==", "Low"),
        ],
        diagnosis=DX_PASSIVE,
        recommendation=(
            "Raise the midfield line by 8-10 m and trigger the press on backward passes."
        ),
        severity="Medium",
        rationale=(
            "A low midfield line plus low pressing invites sustained opponent "
            "possession near our box and gives away territory."
        ),
        priority=10,
        source="Extended",
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
    evidence: list[tuple[Condition, Any]]   # (condition, actual value) pairs
    recommendation: str                     # placeholders already filled in


@dataclass
class InferenceResult:
    working_memory: dict[str, Any]
    origin: dict[str, str]                  # fact -> "Input" | "Abstracted" | "Inferred by Rx"
    fired: list[FiredRule]
    not_fired: list[tuple[Rule, list[str]]] # (rule, reasons it did not fire)
    trace: list[str] = field(default_factory=list)


class InferenceEngine:
    """
    Forward-chaining (data-driven) inference engine.

    Algorithm
    ---------
    1. ABSTRACT raw numeric measurements into symbolic facts (e.g. PPDA 14 -> "Low").
    2. Repeat:
         a. MATCH   : build the agenda = rules not yet fired whose IF-part is true.
         b. RESOLVE : pick one rule by conflict resolution (highest priority,
                      then knowledge-base order).
         c. FIRE    : assert its diagnosis as a new fact, which may enable other rules.
       until the agenda is empty (fixed point).
    Each rule fires at most once (refraction), which guarantees termination.
    """

    def __init__(self, rules: list[Rule]):
        self.rules = rules

    # -- step 1 -------------------------------------------------------------
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

    # -- main loop ----------------------------------------------------------
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

            # Conflict resolution: highest priority first; sort is stable so ties
            # keep knowledge-base order.
            agenda.sort(key=lambda r: -r.priority)
            trace.append(f"[CYCLE {cycle}] Agenda: {[r.rule_id for r in agenda]} "
                         f"-> selected {agenda[0].rule_id} (highest priority {agenda[0].priority})")
            rule = agenda[0]

            evidence = [(c, c.evaluate(wm)[1]) for c in rule.conditions]
            wm[dx(rule.diagnosis)] = True
            origin[dx(rule.diagnosis)] = f"Inferred by {rule.rule_id}"
            fired.append(FiredRule(rule, cycle, evidence, rule.recommendation.format_map(wm)))
            fired_ids.add(rule.rule_id)
            trace.append(f'[FIRE] {rule.rule_id} "{rule.name}" -> asserted fact {dx(rule.diagnosis)} = True')

        # Explanation facility, "why not?" part
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
# Command-line demo:  python expert_system.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = {
        "Player_Name": "Midfielder #8",
        "Player_Sprint_Fatigue": 91,
        "Ball_Loss_Count": 10,
        "Opponent_Build_Up_Time_s": 6.5,
        "Our_PPDA": 15.0,
        "Space_Between_Lines": 18.0,
        "Opponent_Through_Ball_Count": 5,
        "Midfield_Line_Height_m": 30.0,
        "Substitutes_Available": 2,
    }
    result = InferenceEngine(RULES).run(demo)
    print("\n".join(result.trace), "\n")
    for f in result.fired:
        print(f"{f.rule.rule_id} [{f.rule.severity}] {f.rule.diagnosis}\n   -> {f.recommendation}")
