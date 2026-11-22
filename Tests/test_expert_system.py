"""
Test suite for the football expert system (knowledge base, inference engine,
explanation facility and CSV pipeline).

Run from the project root:   python -m pytest -v
"""

import io

import pandas as pd
import pytest

from expert_system import (
    CSV_COLUMNS, FACT_COLUMNS, RULES, STORY_PLAYER, InferenceEngine, analyze_timeline, cast_fact,
    row_to_facts, sample_csv_bytes, sample_dataframe, squad_snapshot, validate_dataframe,
)
from report import ALERT_COLUMNS, alerts_dataframe, build_pdf_report, safe

ENGINE = InferenceEngine(RULES)

# A "quiet" match state that triggers NO rule. Tests override only what they need.
QUIET = dict(
    Player_Name="Test Player", Player_Position="Playmaker", Player_Sprint_Fatigue=40,
    Pass_Accuracy_Under_Press=85, Distance_Covered_km=6.0, Ball_Loss_Count=2,
    Opponent_Build_Up_Time_s=14.0, Our_PPDA=9.0, Space_Between_Lines=11.0,
    Opponent_Through_Ball_Count=0, Opponent_xG_Created=0.05, Midfield_Line_Height_m=50.0,
    Substitutes_Available=3,
)
PERFECT_STORM = dict(
    Player_Sprint_Fatigue=92, Pass_Accuracy_Under_Press=58, Distance_Covered_km=12.1, Ball_Loss_Count=10,
    Opponent_Build_Up_Time_s=6.0, Our_PPDA=15.0, Space_Between_Lines=18.0, Opponent_Through_Ball_Count=5,
    Opponent_xG_Created=1.10, Midfield_Line_Height_m=30.0, Substitutes_Available=2,
)


def run(**overrides):
    return ENGINE.run({**QUIET, **overrides})


def fired(res):
    return {f.rule.rule_id for f in res.fired}


def active(res):
    return {f.rule.rule_id for f in res.active}


def test_quiet_baseline_triggers_nothing():
    assert fired(run()) == set()


# ---------------------------------------------------------------------------
# 1. Every rule: one case where it fires, one where it must stay silent
# ---------------------------------------------------------------------------
R1 = dict(Opponent_Build_Up_Time_s=6.5, Our_PPDA=14.0)
R3 = dict(Space_Between_Lines=18.0, Opponent_Through_Ball_Count=5)
R2 = dict(Player_Sprint_Fatigue=90, Ball_Loss_Count=9)

CASES = {
    "R1": (R1, dict(Opponent_Build_Up_Time_s=6.5, Our_PPDA=9.0)),
    "R2": (R2, dict(Player_Sprint_Fatigue=90, Ball_Loss_Count=3)),
    "R3": (R3, dict(Space_Between_Lines=18.0, Opponent_Through_Ball_Count=1)),
    "R4": ({**R2, "Substitutes_Available": 2}, {**R2, "Substitutes_Available": 0}),
    "R5": ({**R1, **R3}, R1),
    "R6": (dict(Midfield_Line_Height_m=30.0, Our_PPDA=14.0), dict(Midfield_Line_Height_m=45.0, Our_PPDA=14.0)),
    "R7": (dict(Pass_Accuracy_Under_Press=60, Ball_Loss_Count=9), dict(Pass_Accuracy_Under_Press=75, Ball_Loss_Count=9)),
    "R8": ({**R3, "Opponent_xG_Created": 0.9}, dict(Opponent_xG_Created=0.9)),
    "R9": (dict(Distance_Covered_km=12.0, Player_Sprint_Fatigue=82), dict(Distance_Covered_km=10.0, Player_Sprint_Fatigue=82)),
    "R10": (dict(Our_PPDA=7.0, Space_Between_Lines=17.0), dict(Our_PPDA=7.0, Space_Between_Lines=14.0)),
}


def test_every_rule_has_a_test_case():
    assert set(CASES) == {r.rule_id for r in RULES}


@pytest.mark.parametrize("rule_id", CASES)
def test_rule_fires_when_conditions_hold(rule_id):
    assert rule_id in fired(run(**CASES[rule_id][0]))


@pytest.mark.parametrize("rule_id", CASES)
def test_rule_silent_when_a_condition_fails(rule_id):
    assert rule_id not in fired(run(**CASES[rule_id][1]))


# ---------------------------------------------------------------------------
# 2. Boundary values (the thresholds in the rules are strict > / <)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rule_id, overrides, expected", [
    ("R2", {**R2, "Player_Sprint_Fatigue": 85}, False),
    ("R2", {**R2, "Player_Sprint_Fatigue": 86}, True),
    ("R3", {**R3, "Space_Between_Lines": 15.0}, False),
    ("R3", {**R3, "Space_Between_Lines": 15.5}, True),
    ("R7", dict(Pass_Accuracy_Under_Press=70, Ball_Loss_Count=9), False),
    ("R7", dict(Pass_Accuracy_Under_Press=69, Ball_Loss_Count=9), True),
    ("R8", {**R3, "Opponent_xG_Created": 0.75}, False),
    ("R8", {**R3, "Opponent_xG_Created": 0.76}, True),
    ("R9", dict(Distance_Covered_km=11.5, Player_Sprint_Fatigue=90), False),
    ("R9", dict(Distance_Covered_km=11.6, Player_Sprint_Fatigue=90), True),
    ("R9", dict(Distance_Covered_km=12.0, Player_Sprint_Fatigue=80), False),
    ("R9", dict(Distance_Covered_km=12.0, Player_Sprint_Fatigue=81), True),
    ("R10", dict(Our_PPDA=7.0, Space_Between_Lines=16.0), False),
    ("R10", dict(Our_PPDA=7.0, Space_Between_Lines=16.5), True),
])
def test_rule_boundaries(rule_id, overrides, expected):
    assert (rule_id in fired(run(**overrides))) is expected


@pytest.mark.parametrize("overrides, fact, label", [
    (dict(Our_PPDA=8.0), "Our_Pressing_Intensity", "High"),
    (dict(Our_PPDA=8.5), "Our_Pressing_Intensity", "Medium"),
    (dict(Our_PPDA=12.0), "Our_Pressing_Intensity", "Medium"),
    (dict(Our_PPDA=12.5), "Our_Pressing_Intensity", "Low"),
    (dict(Opponent_Build_Up_Time_s=8.0), "Opponent_Build_Up_Speed", "Fast"),
    (dict(Opponent_Build_Up_Time_s=8.5), "Opponent_Build_Up_Speed", "Medium"),
    (dict(Opponent_Build_Up_Time_s=15.5), "Opponent_Build_Up_Speed", "Slow"),
    (dict(Ball_Loss_Count=7), "Ball_Loss_Turnovers", "Medium"),
    (dict(Ball_Loss_Count=8), "Ball_Loss_Turnovers", "High"),
    (dict(Ball_Loss_Count=3), "Ball_Loss_Turnovers", "Low"),
    (dict(Opponent_Through_Ball_Count=3), "Opponent_Through_Balls", "Occasional"),
    (dict(Opponent_Through_Ball_Count=4), "Opponent_Through_Balls", "Frequent"),
    (dict(Midfield_Line_Height_m=34.5), "Midfield_Line_Height", "Low"),
    (dict(Midfield_Line_Height_m=35.0), "Midfield_Line_Height", "Medium"),
    (dict(Midfield_Line_Height_m=50.5), "Midfield_Line_Height", "High"),
    (dict(Opponent_xG_Created=0.8), "Opponent_Threat_Level", "High"),
])
def test_abstraction_thresholds(overrides, fact, label):
    assert run(**overrides).working_memory[fact] == label


# ---------------------------------------------------------------------------
# 3. Forward chaining and conflict resolution
# ---------------------------------------------------------------------------
def test_chained_rule_needs_its_premise():
    """High xG alone must NOT fire R8: it also needs the R3 diagnosis (chaining)."""
    assert "R8" not in fired(run(Opponent_xG_Created=1.5))


def test_chained_rule_fires_in_a_later_cycle_than_its_premise():
    res = run(**R3, Opponent_xG_Created=0.9)
    cycle = {f.rule.rule_id: f.cycle for f in res.fired}
    assert cycle["R3"] < cycle["R8"]


def test_each_rule_fires_at_most_once():
    ids = [f.rule.rule_id for f in run(**PERFECT_STORM).fired]
    assert len(ids) == len(set(ids))


def test_highest_priority_rule_fires_first():
    res = run(**PERFECT_STORM)
    assert res.fired[0].rule.rule_id == "R2"          # priority 40 beats R1/R3/R6/R7/R9
    cycles = [f.cycle for f in res.fired]
    assert cycles == sorted(cycles)


def test_diagnosis_facts_are_asserted_in_working_memory():
    res = run(**R1)
    assert res.working_memory["Diagnosis::Ineffective Pressing Block"] is True
    assert res.origin["Diagnosis::Ineffective Pressing Block"] == "Inferred by R1"


def test_trace_reports_fixed_point():
    assert any("Inference complete" in line for line in run(**PERFECT_STORM).trace)


def test_broader_rule_supersedes_the_rules_it_merges():
    res = run(**PERFECT_STORM)
    by_id = {f.rule.rule_id: f for f in res.fired}
    assert by_id["R1"].superseded_by == "R5"
    assert by_id["R3"].superseded_by == "R5"
    assert by_id["R2"].superseded_by == "R4"
    assert {"R1", "R2", "R3"}.isdisjoint(active(res))
    assert {"R4", "R5", "R8"} <= active(res)


def test_no_supersession_when_rule_stands_alone():
    res = run(**R1)
    assert active(res) == fired(res) == {"R1"}


def test_superseded_diagnoses_still_feed_chained_rules():
    """R3 is merged into R5, but R8 (which chains on R3) must still fire."""
    res = run(**PERFECT_STORM)
    assert "R8" in fired(res)


# ---------------------------------------------------------------------------
# 4. Explanation facility
# ---------------------------------------------------------------------------
def test_evidence_records_actual_values():
    f = next(f for f in run(**R2).fired if f.rule.rule_id == "R2")
    assert [actual for _, actual in f.evidence] == [90, "High"]


def test_recommendation_is_personalised():
    f = next(f for f in run(**R2).fired if f.rule.rule_id == "R2")
    assert "Test Player" in f.recommendation


def test_every_unfired_rule_explains_why():
    res = run(**R1)
    assert len(res.fired) + len(res.not_fired) == len(RULES)
    assert all(reasons for _, reasons in res.not_fired)


# ---------------------------------------------------------------------------
# 5. Sample data regression (the demo match tells a known story)
# ---------------------------------------------------------------------------
EXPECTED_FIRED = {
    15: set(), 30: set(), 45: {"R10"}, 60: {"R1"},
    75: {"R1", "R3", "R5", "R6", "R7", "R8"},
    90: {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"},
}
EXPECTED_ACTIVE = {
    15: set(), 30: set(), 45: {"R10"}, 60: {"R1"},
    75: {"R5", "R6", "R7", "R8"},
    90: {"R4", "R5", "R6", "R7", "R8", "R9"},
}


def story_frame():
    df = sample_dataframe()
    return df[df["Player_Name"] == STORY_PLAYER].reset_index(drop=True)


def test_demo_match_story():
    df = story_frame()
    results = analyze_timeline(df, ENGINE)
    assert len(results) == len(df) == 6
    for minute, res in zip(df["Minute"], results):
        assert fired(res) == EXPECTED_FIRED[minute], f"fired mismatch at minute {minute}"
        assert active(res) == EXPECTED_ACTIVE[minute], f"active mismatch at minute {minute}"


# ---------------------------------------------------------------------------
# 6. Data pipeline: types and CSV validation
# ---------------------------------------------------------------------------
def test_cast_fact_types():
    assert cast_fact("Player_Sprint_Fatigue", 91.0) == 91 and isinstance(cast_fact("Player_Sprint_Fatigue", 91.0), int)
    assert isinstance(cast_fact("Opponent_xG_Created", 1), float)
    assert cast_fact("Player_Name", 7) == "7"


def test_row_to_facts_has_every_fact():
    assert set(row_to_facts(sample_dataframe().iloc[0])) == set(FACT_COLUMNS)


def _csv(mutate=None, drop=None):
    """Build a DataFrame the way a real upload arrives: parsed from CSV text."""
    df = sample_dataframe()
    if drop:
        df = df.drop(columns=drop)
    if mutate:
        df = df.astype(object)
        mutate(df)
    return pd.read_csv(io.StringIO(df.to_csv(index=False)))


def test_sample_csv_round_trip_is_clean():
    df, errors, warnings = validate_dataframe(pd.read_csv(io.BytesIO(sample_csv_bytes())))
    assert errors == [] and warnings == [] and len(df) == 30 and df["Player_Name"].nunique() == 5
    assert list(df.columns) == CSV_COLUMNS


def test_missing_columns_are_reported():
    df, errors, _ = validate_dataframe(pd.DataFrame({"Minute": [15]}))
    assert df is None and "Missing required column" in errors[0]


def test_missing_minute_column_is_assumed():
    df, _, warnings = validate_dataframe(_csv(drop=["Minute"]))
    for _, rows in df.groupby("Player_Name"):                     # assumed per player, not across the whole file
        assert sorted(rows["Minute"]) == [15, 30, 45, 60, 75, 90]
    assert any("Minute" in w for w in warnings)


def test_out_of_range_values_are_clipped():
    def mutate(d): d.loc[1, "Player_Sprint_Fatigue"] = 140
    df, _, warnings = validate_dataframe(_csv(mutate))
    assert df["Player_Sprint_Fatigue"].max() == 100
    assert any("clipped" in w for w in warnings)


def test_unknown_position_falls_back():
    def mutate(d): d.loc[0, "Player_Position"] = "Goalie"
    df, _, warnings = validate_dataframe(_csv(mutate))
    assert df.loc[0, "Player_Position"] == "Playmaker"
    assert any("unknown position" in w for w in warnings)


def test_non_numeric_rows_are_dropped():
    def mutate(d): d.loc[2, "Our_PPDA"] = "n/a"
    df, _, warnings = validate_dataframe(_csv(mutate))
    assert len(df) == 29 and any("Dropped 1" in w for w in warnings)


def test_all_invalid_rows_give_an_error():
    def mutate(d): d["Our_PPDA"] = "n/a"
    df, errors, _ = validate_dataframe(_csv(mutate))
    assert df is None and errors


# ---------------------------------------------------------------------------
# 7. Rule scope + squad view
# ---------------------------------------------------------------------------
def test_rule_scopes():
    assert {r.rule_id for r in RULES if r.scope == "Player"} == {"R2", "R4", "R7", "R9"}
    assert all(r.scope in ("Player", "Team") for r in RULES)


def at_minute(minute):
    df = sample_dataframe()
    return df[df["Minute"] == minute]


def test_squad_has_every_player_ranked_by_risk():
    table, results, _ = squad_snapshot(at_minute(90), ENGINE)
    assert len(table) == len(results) == 5
    assert table["Rank"].tolist() == [1, 2, 3, 4, 5]
    assert table["Player"].tolist() == ["K. Daniels", "A. Moreno", "R. Kovac", "T. Brandt", "L. Haas"]
    assert table["Status"].tolist() == ["Critical", "Critical", "High", "High", "OK"]


def test_squad_player_alerts_are_player_scope_only():
    table, _, _ = squad_snapshot(at_minute(90), ENGINE)
    daniels = table.set_index("Player").loc["K. Daniels"]
    assert "Substitution Alert" in daniels["Player alerts"] and "Press-Resistance Failure" in daniels["Player alerts"]
    assert "Central Corridor" not in " ".join(table["Player alerts"])          # team diagnoses are not repeated per player
    assert table.set_index("Player").loc["L. Haas", "Player alerts"] == "—"


def test_squad_team_alerts_are_listed_once():
    _, _, team = squad_snapshot(at_minute(90), ENGINE)
    ids = [f.rule.rule_id for f in team]
    assert set(ids) == {"R5", "R6", "R8"} and len(ids) == len(set(ids))      # R1/R3 are merged into R5


def test_squad_all_clear_early_in_the_match():
    table, _, team = squad_snapshot(at_minute(15), ENGINE)
    assert set(table["Status"]) == {"OK"} and team == []


# ---------------------------------------------------------------------------
# 8. Export: alerts CSV and PDF report
# ---------------------------------------------------------------------------
def test_alerts_dataframe_structure():
    df = story_frame()
    out = alerts_dataframe(df.to_dict("records"), analyze_timeline(df, ENGINE))
    assert list(out.columns) == ALERT_COLUMNS and len(out) == sum(len(EXPECTED_FIRED[m]) for m in EXPECTED_FIRED)
    r1_at_75 = out[(out["Minute"] == 75) & (out["Rule"] == "R1")].iloc[0]
    assert r1_at_75["Status"] == "Merged into R5" and r1_at_75["Chained"] == "no"
    assert out[(out["Minute"] == 75) & (out["Rule"] == "R8")].iloc[0]["Chained"] == "yes"
    assert set(out["Scope"]) == {"Player", "Team"}


def test_alerts_dataframe_is_empty_but_well_formed_when_nothing_fires():
    out = alerts_dataframe([{}], [run()])
    assert out.empty and list(out.columns) == ALERT_COLUMNS


def _timeline_summary(results):
    return pd.DataFrame([{"Minute": m, "Fatigue %": 50, "Pass acc. %": 80, "Distance km": 5.0, "Ball losses": 2,
                          "Opp. xG": 0.2, "Alerts": len(r.active), "Worst": None, "Diagnoses": "—"}
                         for m, r in zip(EXPECTED_FIRED, results)])


def test_pdf_report_full_and_minimal():
    df = story_frame()
    results = analyze_timeline(df, ENGINE)
    squad, _, team = squad_snapshot(at_minute(90), ENGINE)
    full = build_pdf_report(results[-1], source="Demo match", minute=90, timeline=_timeline_summary(results),
                            squad=squad, team_alerts=team)
    minimal = build_pdf_report(run(), source="Manual simulation")                 # no alerts, no tables
    assert full[:5] == minimal[:5] == b"%PDF-"
    assert len(full) > len(minimal) > 5000


def test_pdf_survives_non_latin_text():
    res = run(Player_Name="أحمد", **R2)
    assert build_pdf_report(res, source="CSV")[:5] == b"%PDF-"
    assert "?" in safe("أحمد") and safe("A — B · C") == "A - B | C"
