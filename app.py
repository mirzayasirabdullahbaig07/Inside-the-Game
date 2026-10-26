import os

import pandas as pd
import streamlit as st

from agents import PERSONAS, llm_available, run_pipeline
from data import SQUADS, generate_match

st.set_page_config(page_title="Inside the Game", page_icon="⚽", layout="wide")

# Allow Streamlit Cloud secrets to configure Azure OpenAI.
for key in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"):
    try:
        if key in st.secrets:
            os.environ.setdefault(key, st.secrets[key])
    except Exception:
        pass  # no secrets file: offline mode


@st.cache_data(show_spinner="Agents are analysing the match…")
def analyse(home, away, seed, persona, use_llm):
    return run_pipeline(generate_match(home, away, seed), persona, use_llm)


st.title("⚽ Inside the Game")
st.caption("Multi-agent match intelligence from synthetic football events. "
           "No real Premier League data is used.")

with st.sidebar:
    st.header("Match setup")
    teams = list(SQUADS)
    home = st.selectbox("Home team", teams, index=0)
    away = st.selectbox("Away team", [t for t in teams if t != home], index=0)
    seed = st.number_input("Match seed", 1, 9999, 7, help="Same seed = same synthetic match.")
    persona = st.radio("Writing voice", list(PERSONAS), index=1)
    ai_ready = llm_available()
    use_llm = st.toggle("Use Azure OpenAI (Foundry)", value=ai_ready, disabled=not ai_ready)
    if not ai_ready:
        st.info("Running offline with template writers. Set the AZURE_OPENAI_* "
                "secrets to enable the LLM.")

state = analyse(home, away, int(seed), persona, use_llm)
stats = state["stats"]
h, a = home, away

c1, c2, c3 = st.columns([2, 1, 2])
c1.metric(h, stats[h].get("goals", 0))
c2.markdown(f"<h2 style='text-align:center'>{state['final_score']}</h2>", unsafe_allow_html=True)
c3.metric(a, stats[a].get("goals", 0))

tab_recap, tab_live, tab_stats, tab_trace = st.tabs(
    ["📝 Recap", "🎙️ Live narrative", "📊 Match intelligence", "🔍 Agent trace"])

with tab_recap:
    st.subheader(f"Full-time recap · {persona} voice")
    st.write(state["recap"])
    verified = state["trace"][-1]["status"] == "ok"
    if verified:
        st.success("Verified by the Checker agent against the match data.")
    else:
        st.warning("Checker flagged an issue. See the Agent trace tab.")
    st.download_button("Download recap", state["recap"], file_name="recap.txt")

with tab_live:
    last = max(e["minute"] for e in state["match"]["events"])
    minute = st.slider("Match minute", 1, last, last)
    shown = [m for m in state["moments"] if m["minute"] <= minute]
    if not shown:
        st.write("Nothing major yet. Drag the slider forward.")
    for m in reversed(shown):
        icon = {"goal": "⚽", "red_card": "🟥", "big_chance": "😬"}[m["kind"]]
        st.markdown(f"**{m['minute']}'** {icon} {m['commentary']}")
        with st.expander("Why this matters"):
            st.write(m["why"])

with tab_stats:
    df = pd.DataFrame({
        h: [stats[h].get(k, 0) for k in ("shots", "on_target", "xg", "pass_into_box", "corner", "yellows", "reds")] + [f"{stats[h]['territory']}%"],
        a: [stats[a].get(k, 0) for k in ("shots", "on_target", "xg", "pass_into_box", "corner", "yellows", "reds")] + [f"{stats[a]['territory']}%"],
    }, index=["Shots", "On target", "xG", "Passes into box", "Corners", "Yellow cards", "Red cards", "Territorial share"]).astype(str)
    st.table(df)
    st.subheader("Momentum")
    st.caption(f"Above zero: {h} on top. Below zero: {a} on top. Measured per 5-minute window.")
    mom = pd.DataFrame(state["momentum"]).set_index("window")
    st.bar_chart(mom)

with tab_trace:
    st.write("Each agent has one job and passes its output to the next. "
             "If the Checker rejects a draft, feedback goes back to the Recap agent. "
             "If the LLM fails, a template recovers.")
    icons = {"ok": "✅", "rejected": "❌", "recovered": "🛟", "error": "⚠️"}
    st.dataframe(pd.DataFrame([{"": icons.get(t["status"], ""), **t} for t in state["trace"]])
                 .drop(columns=["status"]), hide_index=True)