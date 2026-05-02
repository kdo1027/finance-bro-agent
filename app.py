"""
app.py — Streamlit web UI for the Finance Bro Agent.

Run with: streamlit run app.py
"""

import json
import os
import sys

# Guard: this project requires Python 3.10+ and the project venv.
# Run via:  .venv/bin/streamlit run app.py   OR   ./run.sh
if sys.version_info < (3, 10):
    print(
        f"\nERROR: Python {sys.version_info.major}.{sys.version_info.minor} detected.\n"
        "Finance Bro requires Python 3.10+.\n"
        "Use the project venv:  .venv/bin/streamlit run app.py\n"
        "Or activate it first:  source .venv/bin/activate && streamlit run app.py\n"
    )
    sys.exit(1)

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent import build_graph
from schema import UserProfile

st.set_page_config(
    page_title="Finance Bro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Menu definitions ───────────────────────────────────────────────────────────

PHASE_MENUS = {
    "experience": [
        {
            "question": "Q1 — How would you rate your investing experience?",
            "options": [
                ("Just starting out — never invested before", "beginner"),
                ("Tried it a bit — bought stocks or ETFs once or twice", "beginner"),
                ("Some experience — invested regularly for 1–3 years", "intermediate"),
                ("Pretty experienced — active investor, 3+ years", "advanced"),
                ("Very experienced — options, futures, complex strategies", "advanced"),
            ],
            "key": "experience_level",
            "multi": False,
        },
        {
            "question": "Q2 — What have you invested in before? (select all that apply)",
            "options": [
                ("Individual stocks", "individual_stocks"),
                ("ETFs or mutual funds", "etfs_or_mutual_funds"),
                ("Bonds", "bonds"),
                ("Crypto", "crypto"),
                ("Options or futures", "options_or_futures"),
                ("Nothing yet", "none_yet"),
            ],
            "key": "prior_investments",
            "multi": True,
        },
    ],
    "risk": [
        {
            "question": "Q3 — If your portfolio dropped 25% in one month, what would you do?",
            "options": [
                ("Buy more — great buying opportunity", "buy_more"),
                ("Hold — stay the course", "hold"),
                ("Sell some — reduce my exposure", "sell_some"),
                ("Sell everything — get out now", "sell_everything"),
            ],
            "key": "drawdown_response",
            "multi": False,
        },
        {
            "question": "Q4 — Which statement fits you best?",
            "options": [
                ("I want to preserve what I have, even if returns are modest", "preserve_capital"),
                ("I want steady growth with limited downside", "steady_growth"),
                ("I'm comfortable with swings for better long-term returns", "comfortable_with_swings"),
                ("Maximum growth — I can handle big drops", "maximum_growth"),
            ],
            "key": "risk_statement",
            "multi": False,
        },
    ],
    "goals": [
        {
            "question": "Q5 — What's your main reason for investing?",
            "options": [
                ("Retirement", "retirement"),
                ("House down payment", "house_down_payment"),
                ("General wealth building", "general_wealth_building"),
                ("Short-term income", "short_term_income"),
                ("Other", "other"),
            ],
            "key": "investing_goal",
            "multi": False,
        },
        {
            "question": "Q6 — When do you expect to need this money?",
            "options": [
                ("Less than 1 year", "less_than_1_year"),
                ("1 to 3 years", "1_to_3_years"),
                ("3 to 10 years", "3_to_10_years"),
                ("10+ years", "10_plus_years"),
            ],
            "key": "time_horizon",
            "multi": False,
        },
        {
            "question": "Q7 — What's your target annual return?",
            "options": [
                ("Under 5% — keep it safe", "under_5_percent"),
                ("5 to 10% — steady growth", "5_to_10_percent"),
                ("10 to 15% — ambitious", "10_to_15_percent"),
                ("15%+ — aggressive", "15_plus_percent"),
            ],
            "key": "target_return",
            "multi": False,
        },
    ],
    "constraints": [
        {
            "question": "Q8 — Any sector preferences or things to avoid?",
            "options": [
                ("ESG / sustainability focus", "esg_sustainability"),
                ("Exclude tobacco and defense", "exclude_tobacco_defense"),
                ("I have a specific industry in mind", "specific_industry_interest"),
                ("No preference", "no_preference"),
            ],
            "key": "sector_preference",
            "multi": False,
        },
        {
            "question": "Q9 — What percentage of your total savings is this investment?",
            "options": [
                ("Under 10%", "under_10_percent"),
                ("10 to 25%", "10_to_25_percent"),
                ("25 to 50%", "25_to_50_percent"),
                ("50% or more", "50_plus_percent"),
            ],
            "key": "portfolio_percent",
            "multi": False,
        },
    ],
}

PHASE_TO_NODE = {
    "experience":  "experience",
    "risk":        "risk",
    "goals":       "goals",
    "constraints": "constraints",
    "motivation":  "motivation",
    "confirm":     "confirm",
}

PHASE_ORDER = [
    "greet", "experience", "risk", "goals",
    "constraints", "motivation", "confirm", "analyze", "done",
]
PHASE_LABELS = {
    "greet":       "Welcome",
    "experience":  "Experience",
    "risk":        "Risk Tolerance",
    "goals":       "Goals",
    "constraints": "Constraints",
    "motivation":  "Motivation",
    "confirm":     "Confirm Profile",
    "analyze":     "Analyzing...",
    "done":        "Recommendation",
}


# ── Session state ──────────────────────────────────────────────────────────────

def _init():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.graph = None
        st.session_state.graph_state = None
        st.session_state.chat_history = []
        st.session_state.phase = "greet"
        st.session_state.profile = UserProfile().model_dump()


def _get_graph():
    if st.session_state.graph is None:
        st.session_state.graph = build_graph()
    return st.session_state.graph


def _add_msg(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def _invoke(node_name: str, user_input: str | None = None) -> str:
    """Invoke a graph node, sync session state, return the last AI message text."""
    graph = _get_graph()
    state = st.session_state.graph_state

    if user_input:
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

    result = graph.invoke(state, config={"run_name": node_name})

    st.session_state.graph_state = {
        "messages":      result["messages"],
        "profile":       result.get("profile", state["profile"]),
        "current_phase": result["current_phase"],
        "signals":       result.get("signals", state.get("signals", {})),
    }
    st.session_state.phase = result["current_phase"]
    if result.get("profile"):
        st.session_state.profile = result["profile"]

    ai_msgs = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
    return ai_msgs[-1].content if ai_msgs else ""


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _sidebar():
    with st.sidebar:
        st.title("Your Profile")

        profile = UserProfile(**st.session_state.profile)
        status = profile.completion_status()
        filled = len(status["filled"])
        total = 10
        st.progress(filled / total, text=f"{filled} / {total} fields complete")

        st.divider()
        st.subheader("Steps")
        phase = st.session_state.phase
        current_idx = PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 0
        for p in PHASE_ORDER:
            label = PHASE_LABELS[p]
            idx = PHASE_ORDER.index(p)
            if p == phase:
                st.markdown(f"**→ {label}**")
            elif idx < current_idx:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"○ {label}")

        filled_fields = {k: v for k, v in profile.model_dump(mode="json").items() if v is not None}
        if filled_fields:
            st.divider()
            st.subheader("Answers so far")
            for k, v in filled_fields.items():
                label = k.replace("_", " ").title()
                value = ", ".join(v) if isinstance(v, list) else str(v)
                st.markdown(f"**{label}:** {value}")


# ── Chat history ───────────────────────────────────────────────────────────────

def _chat_history():
    for msg in st.session_state.chat_history:
        role = "assistant" if msg["role"] == "agent" else "user"
        avatar = "📈" if msg["role"] == "agent" else "👤"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])


# ── Phase renderers ────────────────────────────────────────────────────────────

def _menu_phase(phase: str):
    menus = PHASE_MENUS[phase]
    with st.form(key=f"form_{phase}"):
        collected: dict = {}
        for menu in menus:
            labels = [opt[0] for opt in menu["options"]]
            values = [opt[1] for opt in menu["options"]]
            st.markdown(f"**{menu['question']}**")
            if menu["multi"]:
                selected = st.multiselect(
                    "Select all that apply:",
                    options=labels,
                    key=f"w_{menu['key']}",
                )
                collected[menu["key"]] = [values[labels.index(s)] for s in selected]
            else:
                selected = st.radio(
                    "Choose one:",
                    options=labels,
                    key=f"w_{menu['key']}",
                    index=None,
                )
                collected[menu["key"]] = values[labels.index(selected)] if selected else None
            st.markdown("")

        sector_detail_input = ""
        if phase == "constraints":
            sector_detail_input = st.text_input(
                "Specific industry in mind? (optional — only fill if you selected the option above)",
                key="w_sector_detail",
            )

        submitted = st.form_submit_button("Continue →", use_container_width=True, type="primary")

    if not submitted:
        return

    # Validate
    for menu in menus:
        if not menu["multi"] and not collected.get(menu["key"]):
            st.warning(f"Please answer: {menu['question']}")
            return
        if menu["multi"] and not collected.get(menu["key"]):
            st.warning(f"Please select at least one option for: {menu['question']}")
            return

    if sector_detail_input and sector_detail_input.strip():
        collected["sector_detail"] = sector_detail_input.strip()

    # Update profile directly (no LLM needed for structured menus)
    profile = UserProfile.model_validate({**st.session_state.profile, **collected})
    st.session_state.profile = profile.model_dump()
    st.session_state.graph_state["profile"] = profile.model_dump()

    parts = [f"{k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in collected.items()]
    user_msg = ". ".join(parts)
    _add_msg("user", user_msg)

    with st.spinner("Processing..."):
        response = _invoke(PHASE_TO_NODE[phase], user_msg)
    if response:
        _add_msg("agent", response)

    st.rerun()


def _motivation_phase():
    with st.form(key="form_motivation"):
        st.markdown("**Q10 — What's motivating you to invest right now?**")
        text = st.text_area(
            "Tell us in your own words:",
            placeholder="e.g., I want to build a safety net for my family and grow wealth over the next decade...",
            height=120,
            key="w_motivation",
        )
        submitted = st.form_submit_button("Submit →", use_container_width=True, type="primary")

    if not submitted:
        return

    if not text.strip():
        st.warning("Please share something about your motivation before continuing.")
        return

    _add_msg("user", text.strip())
    with st.spinner("Processing your response..."):
        response = _invoke("motivation", text.strip())
    if response:
        _add_msg("agent", response)
    st.rerun()


def _confirm_phase():
    st.markdown("**Does everything above look correct?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes — run the analysis", use_container_width=True, type="primary"):
            _add_msg("user", "yes")
            with st.spinner("Confirming..."):
                response = _invoke("confirm", "yes")
            if response:
                _add_msg("agent", response)
            st.rerun()
    with col2:
        if st.button("↩ Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _analyze_phase():
    """Auto-trigger analyze (which chains into synthesize) with a progress indicator."""
    with st.spinner("🔍 Analyzing market signals and generating your recommendation — this takes about a minute..."):
        response = _invoke("analyze")

    if response:
        _add_msg("agent", response)

    profile = UserProfile(**st.session_state.profile)
    with open("user_profile.json", "w") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2)

    st.session_state.phase = "done"
    st.session_state.graph_state["current_phase"] = "done"
    st.rerun()


def _done_phase():
    profile = UserProfile(**st.session_state.profile)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Profile JSON",
            data=json.dumps(profile.model_dump(mode="json"), indent=2),
            file_name="finance_profile.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        if st.button("🔄 Start Over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _init()

    st.title("📈 Finance Bro")
    st.caption("Your personalized AI investment advisor")

    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not set. Add it to your .env file and restart.")
        st.stop()

    _sidebar()

    # First-run: build graph and invoke the greet node
    if not st.session_state.initialized:
        graph = _get_graph()
        init_state = {
            "messages":      [],
            "profile":       UserProfile().model_dump(),
            "current_phase": "greet",
            "signals":       {},
        }
        with st.spinner("Starting up..."):
            result = graph.invoke(init_state, config={"run_name": "greet"})

        st.session_state.graph_state = {
            "messages":      result["messages"],
            "profile":       result.get("profile", init_state["profile"]),
            "current_phase": result["current_phase"],
            "signals":       {},
        }
        st.session_state.phase = result["current_phase"]

        ai_msgs = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_msgs:
            _add_msg("agent", ai_msgs[-1].content)

        st.session_state.initialized = True
        st.rerun()

    _chat_history()

    phase = st.session_state.phase

    if phase in PHASE_MENUS:
        _menu_phase(phase)
    elif phase == "motivation":
        _motivation_phase()
    elif phase == "confirm":
        _confirm_phase()
    elif phase in ("analyze", "synthesize"):
        _analyze_phase()
    elif phase == "done":
        _done_phase()


if __name__ == "__main__":
    main()
