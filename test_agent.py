"""
test_agent.py — Smoke tests for Agentic Finance Bro.

Run with:
    python test_agent.py

Two modes:
  - Unit tests (always run, no API key needed): schema models, tool outputs
  - End-to-end test (runs only if OPENAI_API_KEY is set): pumps hardcoded
    answers through the full LangGraph pipeline and prints the recommendation

Usage:
    python test_agent.py          # run everything
    python test_agent.py --units  # unit tests only (no API key needed)
"""

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Unit tests ─────────────────────────────────────────────────────────────

def test_schema_defaults():
    from schema import UserProfile
    p = UserProfile()
    assert p.experience_level is None
    assert p.prior_investments is None
    status = p.completion_status()
    assert len(status["filled"]) == 0
    assert "experience_level" in status["missing"]
    print("  [PASS] UserProfile defaults and completion_status()")


def test_schema_roundtrip():
    from schema import (
        DrawdownResponse, ExperienceLevel, InvestingGoal, InvestmentType,
        PortfolioPercent, RiskStatement, SectorPreference, TargetReturn,
        TimeHorizon, UserProfile,
    )
    p = UserProfile(
        experience_level=ExperienceLevel.INTERMEDIATE,
        prior_investments=[InvestmentType.STOCKS, InvestmentType.ETFS],
        drawdown_response=DrawdownResponse.HOLD,
        risk_statement=RiskStatement.STEADY,
        investing_goal=InvestingGoal.WEALTH,
        time_horizon=TimeHorizon.THREE_TO_TEN,
        target_return=TargetReturn.FIVE_TO_TEN,
        sector_preference=SectorPreference.NO_PREFERENCE,
        portfolio_percent=PortfolioPercent.TEN_TO_25,
        investing_motivation="Building a safety net for my family.",
    )
    assert p.is_complete()
    dumped = p.model_dump(mode="json")
    p2 = UserProfile(**dumped)
    assert p2.experience_level == ExperienceLevel.INTERMEDIATE
    assert p2.prior_investments == [InvestmentType.STOCKS, InvestmentType.ETFS]
    print("  [PASS] UserProfile roundtrip serialization")


def test_tools():
    from tools import ALL_TOOLS, get_fundamentals_signal, get_macro_signal, get_momentum_signal, get_sentiment_signal

    sectors = ["tech", "healthcare"]

    result = get_sentiment_signal.invoke({"sectors": sectors})
    assert "tech" in result
    assert "sentiment_score" in result["tech"]
    assert result["tech"]["label"] in ("positive", "negative", "neutral")

    result = get_fundamentals_signal.invoke({"sectors": sectors})
    assert "healthcare" in result
    assert "pe_ratio" in result["healthcare"]

    result = get_macro_signal.invoke({})
    assert "fed_funds_rate" in result
    assert "cpi_yoy" in result

    result = get_momentum_signal.invoke({"sectors": sectors})
    assert "tech" in result
    assert result["tech"]["trend"] in ("uptrend", "downtrend", "flat")

    assert len(ALL_TOOLS) == 4
    print("  [PASS] All four signal tools return expected structure")


def test_graph_builds():
    from agent import build_graph
    graph = build_graph()
    assert graph is not None
    print("  [PASS] LangGraph compiles without errors")


def run_unit_tests():
    print("\n── Unit Tests ────────────────────────────────────────────────")
    failures = []
    for fn in [test_schema_defaults, test_schema_roundtrip, test_tools, test_graph_builds]:
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
    if failures:
        print(f"\n{len(failures)} test(s) failed: {', '.join(failures)}")
        return False
    print("\nAll unit tests passed.")
    return True


# ── End-to-end test ────────────────────────────────────────────────────────

# Hardcoded answers for each intake phase — one per graph node
SCRIPTED_ANSWERS = {
    "experience": "I'd say a 3. I've bought some ETFs and individual stocks, but never touched options or crypto.",
    "risk":       "If my portfolio dropped 25% I'd hold and maybe buy a bit more. I think the third statement fits — comfortable with swings for better long-term returns.",
    "goals":      "My main goal is general wealth building. Time horizon is 3-10 years. I'm aiming for 5-10% annualized.",
    "constraints":"No real sector restrictions, no preference. This represents about 10-25% of my total savings.",
    "motivation": "I want to make my money work harder than a savings account while I'm still young.",
    "confirm":    "Yes, that all looks correct. Let's go.",
}


def run_e2e_test():
    print("\n── End-to-End Test ───────────────────────────────────────────")
    print("Pumping scripted answers through the full pipeline...\n")

    from agent import build_graph
    from langchain_core.messages import HumanMessage
    from schema import UserProfile

    graph = build_graph()

    state = {
        "messages": [],
        "profile": UserProfile().model_dump(),
        "current_phase": "greet",
        "signals": {},
    }

    # Phase map mirrors main.py
    phase_to_node = {
        "experience":  "experience",
        "risk":        "risk",
        "goals":       "goals",
        "constraints": "constraints",
        "motivation":  "motivation",
        "confirm":     "confirm",
    }

    # Step 1: greeting
    print("[greet] Running greeting node...")
    result = graph.invoke(state, config={"run_name": "greet"})
    state.update({"messages": result["messages"], "current_phase": result["current_phase"]})
    last_ai = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
    if last_ai:
        print(f"  Agent: {last_ai[-1].content[:120]}...\n")

    # Step 2: conversation phases
    for phase, answer in SCRIPTED_ANSWERS.items():
        if state["current_phase"] == "done":
            break

        node = phase_to_node.get(state["current_phase"])
        if not node:
            print(f"  Unexpected phase: {state['current_phase']}")
            break

        print(f"[{node}] User: \"{answer[:80]}...\"")
        state["messages"] = list(state["messages"]) + [HumanMessage(content=answer)]

        result = graph.invoke(state, config={"run_name": node})
        state.update({
            "messages": result["messages"],
            "current_phase": result["current_phase"],
            "profile": result.get("profile", state["profile"]),
        })

        last_ai = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if last_ai:
            print(f"  Agent: {last_ai[-1].content[:120]}...\n")

    # Step 3: auto-run analysis + synthesis (same as main.py)
    if state["current_phase"] == "analyze":
        print("[analyze → synthesize] Running market analysis...")
        result = graph.invoke(state, config={"run_name": "analyze"})
        state.update({
            "messages": result["messages"],
            "current_phase": result["current_phase"],
            "signals": result.get("signals", {}),
            "profile": result.get("profile", state["profile"]),
        })
        last_ai = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if last_ai:
            print(f"\n── Recommendation ────────────────────────────────────────────")
            print(last_ai[-1].content)

    # Export profile
    profile = UserProfile(**state["profile"])
    output_path = "user_profile_test.json"
    with open(output_path, "w") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2)
    print(f"\nProfile saved to {output_path}")
    print(f"Profile complete: {profile.is_complete()}")
    print("\nEnd-to-end test passed.")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    units_only = "--units" in sys.argv

    ok = run_unit_tests()
    if not ok:
        sys.exit(1)

    if units_only:
        sys.exit(0)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-gemini-api-key-here":
        print("\nSkipping end-to-end test: GOOGLE_API_KEY not set.")
        print("Set it in .env or export it, then re-run without --units.")
        sys.exit(0)

    run_e2e_test()
