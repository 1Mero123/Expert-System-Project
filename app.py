"""
app.py
======
Streamlit dashboard for the Football Match & Player Performance Expert System.
Supports both Manual Input/Scenarios AND Batch/Timeline CSV Match Data Upload.
"""

import io
import pandas as pd
import streamlit as st

from expert_system import RULES, InferenceEngine

st.set_page_config(page_title="Football Tactical Expert System", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------------------
# Input defaults and demo scenarios
# ---------------------------------------------------------------------------
DEFAULTS = {
    "Player_Name": "Midfielder #8",
    "Player_Sprint_Fatigue": 60,
    "Ball_Loss_Count": 3,
    "Opponent_Build_Up_Time_s": 12.0,
    "Our_PPDA": 9.0,
    "Space_Between_Lines": 11.0,
    "Opponent_Through_Ball_Count": 1,
    "Midfield_Line_Height_m": 45.0,
    "Substitutes_Available": 3,
}

SCENARIOS = {
    "Custom (manual input)": None,
    "Balanced match - no alerts": {
        "Player_Sprint_Fatigue": 55, "Ball_Loss_Count": 2, "Opponent_Build_Up_Time_s": 14.0,
        "Our_PPDA": 7.0, "Space_Between_Lines": 11.0, "Opponent_Through_Ball_Count": 0,
        "Midfield_Line_Height_m": 52.0, "Substitutes_Available": 3},
    "Opponent counter-attacks, our press is weak": {
        "Player_Sprint_Fatigue": 60, "Ball_Loss_Count": 3, "Opponent_Build_Up_Time_s": 6.5,
        "Our_PPDA": 14.0, "Space_Between_Lines": 12.0, "Opponent_Through_Ball_Count": 1,
        "Midfield_Line_Height_m": 44.0, "Substitutes_Available": 3},
    "Tired playmaker losing the ball": {
        "Player_Sprint_Fatigue": 91, "Ball_Loss_Count": 9, "Opponent_Build_Up_Time_s": 12.0,
        "Our_PPDA": 9.0, "Space_Between_Lines": 10.0, "Opponent_Through_Ball_Count": 1,
        "Midfield_Line_Height_m": 45.0, "Substitutes_Available": 2},
    "Open central corridor": {
        "Player_Sprint_Fatigue": 70, "Ball_Loss_Count": 5, "Opponent_Build_Up_Time_s": 11.0,
        "Our_PPDA": 10.0, "Space_Between_Lines": 19.0, "Opponent_Through_Ball_Count": 5,
        "Midfield_Line_Height_m": 40.0, "Substitutes_Available": 3},
    "Perfect storm (everything wrong)": {
        "Player_Sprint_Fatigue": 92, "Ball_Loss_Count": 10, "Opponent_Build_Up_Time_s": 6.0,
        "Our_PPDA": 15.0, "Space_Between_Lines": 18.0, "Opponent_Through_Ball_Count": 5,
        "Midfield_Line_Height_m": 30.0, "Substitutes_Available": 2},
}

# Sample CSV data representing a full 90-minute match split into 15-min intervals
SAMPLE_MATCH_CSV = pd.DataFrame([
    {"Minute_Interval": "00-15 min", "Player_Name": "Midfielder #8", "Player_Sprint_Fatigue": 20, "Ball_Loss_Count": 1, "Opponent_Build_Up_Time_s": 14.0, "Our_PPDA": 7.5, "Space_Between_Lines": 10.0, "Opponent_Through_Ball_Count": 0, "Midfield_Line_Height_m": 52.0, "Substitutes_Available": 5},
    {"Minute_Interval": "15-30 min", "Player_Name": "Midfielder #8", "Player_Sprint_Fatigue": 38, "Ball_Loss_Count": 2, "Opponent_Build_Up_Time_s": 11.5, "Our_PPDA": 8.5, "Space_Between_Lines": 12.0, "Opponent_Through_Ball_Count": 1, "Midfield_Line_Height_m": 48.0, "Substitutes_Available": 5},
    {"Minute_Interval": "30-45 min", "Player_Name": "Right Winger #7", "Player_Sprint_Fatigue": 55, "Ball_Loss_Count": 3, "Opponent_Build_Up_Time_s": 7.0, "Our_PPDA": 13.5, "Space_Between_Lines": 13.0, "Opponent_Through_Ball_Count": 2, "Midfield_Line_Height_m": 43.0, "Substitutes_Available": 5},
    {"Minute_Interval": "45-60 min", "Player_Name": "Center Back #4", "Player_Sprint_Fatigue": 68, "Ball_Loss_Count": 4, "Opponent_Build_Up_Time_s": 9.5, "Our_PPDA": 11.0, "Space_Between_Lines": 17.5, "Opponent_Through_Ball_Count": 4, "Midfield_Line_Height_m": 39.0, "Substitutes_Available": 4},
    {"Minute_Interval": "60-75 min", "Player_Name": "Midfielder #8", "Player_Sprint_Fatigue": 88, "Ball_Loss_Count": 9, "Opponent_Build_Up_Time_s": 10.0, "Our_PPDA": 10.5, "Space_Between_Lines": 14.0, "Opponent_Through_Ball_Count": 2, "Midfield_Line_Height_m": 42.0, "Substitutes_Available": 3},
    {"Minute_Interval": "75-90 min", "Player_Name": "Defensive Mid #6", "Player_Sprint_Fatigue": 93, "Ball_Loss_Count": 10, "Opponent_Build_Up_Time_s": 6.0, "Our_PPDA": 15.5, "Space_Between_Lines": 19.0, "Opponent_Through_Ball_Count": 5, "Midfield_Line_Height_m": 31.0, "Substitutes_Available": 2},
])

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def load_scenario() -> None:
    preset = SCENARIOS[st.session_state["scenario"]]
    if preset:
        st.session_state.update(preset)


# ---------------------------------------------------------------------------
# Sidebar: Data Source Selection (CSV Upload OR Manual Simulation)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Data Source")
    data_mode = st.radio(
        "Choose input mode:",
        ["Manual / Demo Scenarios", "Upload Match CSV File"],
        help="Upload a real match tracking CSV file or simulate manually."
    )
    st.divider()

    uploaded_df = None
    selected_row_idx = 0

    if data_mode == "Upload Match CSV File":
        st.subheader("📤 Match Tracking CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

        # Download template button
        csv_buffer = SAMPLE_MATCH_CSV.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Match CSV",
            data=csv_buffer,
            file_name="sample_match_tracking.csv",
            mime="text/csv",
            help="Download a template CSV file with 90-minute match intervals to test."
        )

        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                # Validate required columns
                missing_cols = [c for c in DEFAULTS.keys() if c not in uploaded_df.columns]
                if missing_cols:
                    st.error(f"Missing columns in CSV: {', '.join(missing_cols)}")
                    uploaded_df = None
                else:
                    label_col = "Minute_Interval" if "Minute_Interval" in uploaded_df.columns else uploaded_df.columns[0]
                    options = [f"Row {i+1}: {row[label_col]} ({row['Player_Name']})" for i, row in uploaded_df.iterrows()]
                    chosen = st.selectbox("Select interval / record to inspect:", options)
                    selected_row_idx = options.index(chosen)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
        else:
            st.info("Upload a CSV file above, or download the sample CSV to test.")

    else:
        st.header("📡 Match Tracking Inputs")
        st.selectbox("Load demo scenario", list(SCENARIOS), key="scenario", on_change=load_scenario)
        st.divider()

        st.subheader("Player")
        st.text_input("Player name / number", key="Player_Name")
        st.slider("Player sprint fatigue (%)", 0, 100, key="Player_Sprint_Fatigue")
        st.slider("Ball-loss turnovers in dangerous zones (per 15 min)", 0, 15, key="Ball_Loss_Count")

        st.subheader("Opponent")
        st.slider("Opponent build-up time to final third (s)", 3.0, 25.0, step=0.5, key="Opponent_Build_Up_Time_s")
        st.slider("Opponent through balls (per 15 min)", 0, 8, key="Opponent_Through_Ball_Count")

        st.subheader("Our team shape")
        st.slider("Our pressing intensity - PPDA (lower = more intense)", 4.0, 20.0, step=0.5, key="Our_PPDA")
        st.slider("Space between defensive & midfield lines (m)", 5.0, 30.0, step=0.5, key="Space_Between_Lines")
        st.slider("Midfield line height (m from own goal)", 20.0, 70.0, step=1.0, key="Midfield_Line_Height_m")
        st.slider("Substitutions still available", 0, 5, key="Substitutes_Available")

# ---------------------------------------------------------------------------
# Prepare Facts & Run the Expert System
# ---------------------------------------------------------------------------
engine = InferenceEngine(RULES)

if data_mode == "Upload Match CSV File" and uploaded_df is not None:
    row_data = uploaded_df.iloc[selected_row_idx].to_dict()
    raw_facts = {key: row_data[key] for key in DEFAULTS}
else:
    raw_facts = {key: st.session_state[key] for key in DEFAULTS}

result = engine.run(raw_facts)
wm = result.working_memory

# ---------------------------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------------------------
st.title("⚽ Advanced Football Match & Player Performance Expert System")
st.caption("Rule-based expert system · forward-chaining inference · full explanation facility")

# If a full CSV match is uploaded, show a Full-Match Timeline Summary first!
if data_mode == "Upload Match CSV File" and uploaded_df is not None:
    with st.expander("📊 Full Match Analysis Report (All CSV Rows)", expanded=True):
        summary_rows = []
        for idx, row in uploaded_df.iterrows():
            r_facts = {key: row[key] for key in DEFAULTS}
            r_res = engine.run(r_facts)
            diagnoses = [f"{f.rule.rule_id}: {f.rule.diagnosis}" for f in r_res.fired]
            recs = [f.recommendation for f in r_res.fired]
            summary_rows.append({
                "Interval / Record": row.get("Minute_Interval", f"Row {idx+1}"),
                "Player": row["Player_Name"],
                "Alerts Triggered": len(r_res.fired),
                "Tactical Diagnoses": " | ".join(diagnoses) if diagnoses else "✅ Normal / No Alerts",
                "Recommendations": " | ".join(recs) if recs else "Keep current setup",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.subheader(f"🔍 Detailed Inspection: {uploaded_df.iloc[selected_row_idx].get('Minute_Interval', f'Row {selected_row_idx+1}')}")

# Row of key (abstracted) facts
cols = st.columns(5)
cols[0].metric("Opponent build-up", wm["Opponent_Build_Up_Speed"], f'{wm["Opponent_Build_Up_Time_s"]:.1f} s')
cols[1].metric("Our pressing", wm["Our_Pressing_Intensity"], f'PPDA {wm["Our_PPDA"]:.1f}', delta_color="off")
cols[2].metric("Ball-loss turnovers", wm["Ball_Loss_Turnovers"], f'{wm["Ball_Loss_Count"]} / 15 min', delta_color="off")
cols[3].metric("Opponent through balls", wm["Opponent_Through_Balls"], f'{wm["Opponent_Through_Ball_Count"]} / 15 min', delta_color="off")
cols[4].metric("Midfield line", wm["Midfield_Line_Height"], f'{wm["Midfield_Line_Height_m"]:.0f} m', delta_color="off")
st.progress(min(int(wm["Player_Sprint_Fatigue"]), 100) / 100, text=f'{wm["Player_Name"]} sprint fatigue: {wm["Player_Sprint_Fatigue"]} %')

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3}
SEVERITY_BOX = {"Critical": st.error, "High": st.warning, "Medium": st.info, "Info": st.success}
fired_sorted = sorted(result.fired, key=lambda f: SEVERITY_ORDER[f.rule.severity])

tab_dx, tab_why, tab_trace, tab_wm, tab_kb = st.tabs(
    ["🩺 Diagnosis & Recommendations", "💡 Explanation (Why?)", "🔁 Inference Trace",
     "🧠 Working Memory", "📚 Knowledge Base"]
)

# --- Tab 1: diagnoses -------------------------------------------------------
with tab_dx:
    if not fired_sorted:
        st.success("✅ No tactical alerts: the current metrics trigger none of the rules.")
    else:
        st.subheader(f"{len(fired_sorted)} alert(s) triggered")
        for f in fired_sorted:
            SEVERITY_BOX[f.rule.severity](
                f"**{f.rule.diagnosis}**  ·  _{f.rule.severity}_  ·  {f.rule.rule_id}\n\n"
                f"➡️ **Recommendation:** {f.recommendation}"
            )

# --- Tab 2: explanation facility -------------------------------------------
with tab_why:
    if not result.fired:
        st.info("No rule fired, so there is nothing to explain yet. See below why each rule did not fire.")
    for f in fired_sorted:
        with st.expander(f"{f.rule.rule_id} - {f.rule.diagnosis}", expanded=True):
            st.markdown(f"**Rule:** {f.rule.name}  ·  **Fired in cycle:** {f.cycle}  ·  **Type:** {f.rule.source}")
            st.code(f.rule.as_text(), language=None, wrap_lines=True)
            st.markdown("**Evidence (why the IF-part was true):**")
            st.dataframe(
                pd.DataFrame(
                    {"Condition": c.describe(),
                     "Actual value": "—" if c.fact.startswith("Diagnosis::") else str(actual),
                     "Satisfied": "✅"}
                    for c, actual in f.evidence
                ),
                hide_index=True,
            )
            st.markdown(f"**Football reasoning:** {f.rule.rationale}")
            if any(c.fact.startswith("Diagnosis::") for c, _ in f.evidence):
                st.caption("🔗 This rule was reached by chaining on conclusions made by earlier rules.")

    st.subheader("Why did the other rules NOT fire?")
    if not result.not_fired:
        st.write("All rules fired.")
    for rule, reasons in result.not_fired:
        with st.expander(f"{rule.rule_id} - {rule.diagnosis} (not triggered)"):
            st.markdown("Unmet condition(s):")
            for reason in reasons:
                st.markdown(f"- ❌ {reason}")

# --- Tab 3: trace -----------------------------------------------------------
with tab_trace:
    st.markdown("Step-by-step log of the forward-chaining cycle (abstraction → match → resolve → fire).")
    st.code("\n".join(result.trace), language=None)

# --- Tab 4: working memory --------------------------------------------------
with tab_wm:
    st.markdown("All facts currently in working memory, and where each one came from.")
    st.dataframe(
        pd.DataFrame({"Fact": k, "Value": str(v), "Origin": result.origin[k]} for k, v in wm.items()),
        hide_index=True,
    )

# --- Tab 5: knowledge base --------------------------------------------------
with tab_kb:
    st.markdown("Production rules in the knowledge base (R1-R3 are the core rules; R4-R6 are extensions).")
    st.dataframe(
        pd.DataFrame(
            {"ID": r.rule_id, "Type": r.source, "Priority": r.priority, "Severity": r.severity, "Rule": r.as_text()}
            for r in RULES
        ),
        hide_index=True,
)