"""
agent.py — The LangGraph agent for Agentic Finance Bro.

This file defines the state machine that:
1. Greets the user and explains the system
2. Walks through intake questions (Q1-Q10)
3. Confirms the profile
4. Calls signal tools (A-D)
5. Synthesizes a recommendation

Architecture:
- StateGraph with nodes connected by edges.
- Each node is a function that reads/writes AgentState.
- The LLM handles natural conversation + structured extraction.
"""

import json
import os
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from schema import ExtractedMotivation, UserProfile
from tools import SECTOR_TOOLS, STOCK_TOOLS, SECTOR_STOCKS

# Agent State

class AgentState(TypedDict):
    """
    The state that flows through every node in the graph.

    - messages: full conversation history (LangGraph auto-appends via add_messages)
    - profile: the user's investment profile being built up
    - current_phase: which intake phase we're in
    - signals: results from tool calls (populated in analysis phase)
    """
    messages: Annotated[list, add_messages]
    profile: dict          # serialized UserProfile
    current_phase: str     # "greet" | "experience" | "risk" | "goals" | "constraints" | "motivation" | "confirm" | "analyze" | "synthesize" | "done"
    signals: dict          # results from Signals A-D


# LLM Setup

def _primary_llm(temperature: float = 0.3):
    """Google Gemini if key is present, otherwise OpenAI."""
    if os.getenv("GOOGLE_API_KEY"):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    raise ValueError("Set GOOGLE_API_KEY or OPENAI_API_KEY in .env")


def _openai_fallback(temperature: float = 0.3):
    """OpenAI fallback — only created when both keys are present."""
    if os.getenv("GOOGLE_API_KEY") and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    return None


def get_llm(temperature: float = 0.3):
    """LLM for plain invoke() calls, with OpenAI fallback on any exception."""
    primary = _primary_llm(temperature)
    fallback = _openai_fallback(temperature)
    if fallback:
        return primary.with_fallbacks([fallback])
    return primary


def get_extraction_llm(schema, temperature: float = 0.3):
    """Structured-output chain with OpenAI fallback."""
    primary = _primary_llm(temperature)
    fallback = _openai_fallback(temperature)
    chain = primary.with_structured_output(schema)
    if fallback:
        return chain.with_fallbacks([fallback.with_structured_output(schema)])
    return chain


def _text(response) -> str:
    """Extract plain text from a response whose content may be a list of blocks."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def _has_data(extracted) -> bool:
    """Return True if the LLM extracted at least one non-None field."""
    return any(v is not None for v in extracted.model_dump().values())


# System Prompt

SYSTEM_PROMPT = """You are Finance Bro, a friendly investment guide for everyday people.

Your rules:
- Never use financial jargon. If a word needs a definition, replace it with plain English.
- Never guarantee returns or hype a stock. Be honest about risk.
- Always tie advice back to what the user told you about themselves.
- Be short. Two sentences is better than five.
- During intake: ask one or two questions at a time, never a long form.
- Don't repeat questions the user already answered.
- During recommendations: always end with one concrete next step the user can actually take."""


# Node Functions

def greet_node(state: AgentState) -> dict:
    """Opening message — introduce the agent and kick off the intake."""
    return {
        "messages": [AIMessage(content=(
            "Hey! I'm Finance Bro — I'll help you figure out which investments make the most sense for you.\n"
            "I'll ask a few quick questions about your experience, risk tolerance, and goals, "
            "then analyze the market and give you a personalized recommendation.\n"
            "Let's start!"
        ))],
        "current_phase": "experience",
    }


def experience_node(state: AgentState) -> dict:
    """Profile already populated by main.py — advance to risk with a hardcoded acknowledgment."""
    return {
        "messages": [AIMessage(content="Got it! Now let's see how you handle risk.")],
        "profile": state["profile"],
        "current_phase": "risk",
    }


def risk_node(state: AgentState) -> dict:
    """Profile already populated by main.py — advance to goals with a hardcoded acknowledgment."""
    return {
        "messages": [AIMessage(content="Perfect. Next up — your goals and time horizon.")],
        "profile": state["profile"],
        "current_phase": "goals",
    }


def goals_node(state: AgentState) -> dict:
    """Profile already populated by main.py — advance to constraints with a hardcoded acknowledgment."""
    return {
        "messages": [AIMessage(content="Nice. Two more quick ones about your preferences.")],
        "profile": state["profile"],
        "current_phase": "constraints",
    }


def constraints_node(state: AgentState) -> dict:
    """Profile already populated by main.py — advance to motivation with a hardcoded prompt."""
    return {
        "messages": [AIMessage(content="Almost there! Last one: in a sentence, why are you investing right now?")],
        "profile": state["profile"],
        "current_phase": "motivation",
    }


_DISPLAY_LABELS = {
    "etfs_or_mutual_funds": "ETFs or Mutual Funds",
    "options_or_futures":   "Options or Futures",
    "none_yet":             "None yet",
    "less_than_1_year":     "Less than 1 year",
    "1_to_3_years":         "1–3 years",
    "3_to_10_years":        "3–10 years",
    "10_plus_years":        "10+ years",
    "under_5_percent":      "Under 5%",
    "5_to_10_percent":      "5–10%",
    "10_to_15_percent":     "10–15%",
    "15_plus_percent":      "15%+",
    "under_10_percent":     "Under 10%",
    "10_to_25_percent":     "10–25%",
    "25_to_50_percent":     "25–50%",
    "50_plus_percent":      "50%+",
    "esg_sustainability":   "ESG / Sustainability",
    "exclude_tobacco_defense": "Exclude tobacco & defense",
    "specific_industry_interest": "Specific industry",
}


def _format_profile_summary(profile: UserProfile) -> str:
    """Build a readable profile summary without an LLM call."""
    def _display(val) -> str:
        if isinstance(val, list):
            return ", ".join(_display(v) for v in val)
        if hasattr(val, "value"):
            raw = val.value
            return _DISPLAY_LABELS.get(raw, raw.replace("_", " ").title())
        return str(val)  # free text — preserve as-is

    status = profile.completion_status()
    lines = ["Here's what I have for you:\n"]
    for field_name, value in status["filled"].items():
        label = field_name.replace("_", " ").title()
        lines.append(f"  {label}: {_display(value)}")
    lines.append("\nDoes everything look right? Type 'yes' to run the analysis, or let me know what to change.")
    return "\n".join(lines)


def motivation_node(state: AgentState) -> dict:
    """Extract free-text motivation (Q10), then show profile summary for confirmation."""
    extraction_llm = get_extraction_llm(ExtractedMotivation)
    user_msg = state["messages"][-1].content

    extracted = extraction_llm.invoke([
        SystemMessage(content="Extract the user's investing motivation from their message."),
        HumanMessage(content=user_msg),
    ])

    profile = UserProfile(**state["profile"])
    if extracted.investing_motivation:
        profile.investing_motivation = extracted.investing_motivation

    if not _has_data(extracted):
        return {
            "messages": [AIMessage(content="I didn't quite catch that. In a sentence or two, why are you investing right now?")],
            "profile": profile.model_dump(),
            "current_phase": "motivation",
        }

    return {
        "messages": [AIMessage(content=_format_profile_summary(profile))],
        "profile": profile.model_dump(),
        "current_phase": "confirm",
    }


def confirm_node(state: AgentState) -> dict:
    """Check if user confirmed. If yes, move to analysis. If not, explain how to correct."""
    user_msg = state["messages"][-1].content.lower()

    confirms = ["yes", "yeah", "yep", "looks good", "correct", "confirm", "let's go", "go ahead", "sure", "that's right"]
    is_confirmed = any(c in user_msg for c in confirms)

    if is_confirmed:
        return {
            "messages": [AIMessage(content="On it! Running market analysis across sentiment, fundamentals, macro, and momentum — this takes a moment.")],
            "current_phase": "analyze",
        }
    else:
        return {
            "messages": [AIMessage(content="To update an answer, type 'quit' and run me again — the menus will let you pick fresh. Or type 'yes' to proceed with your current answers.")],
            "current_phase": "confirm",
        }


def _compute_sector_score(sector: str, sentiment: dict, fundamentals: dict, macro: dict, momentum: dict) -> float:
    """Pre-score a sector to find top candidates for stock-level analysis.
    Weights: fundamentals 35%, sentiment 25%, macro 25%, momentum 15%.
    """
    s = sentiment.get(sector, {}).get("sentiment_score", 0.0)
    f = fundamentals.get(sector, {}).get("health_score", 0.0)
    c = macro.get("macro_score", 0.0)
    m = momentum.get(sector, {}).get("momentum_score", 0.0)
    return (s * 0.25) + (f * 0.35) + (c * 0.25) + (m * 0.15)


def analyze_node(state: AgentState) -> dict:
    """Two-pass analysis: score sectors first, then fetch stock-level signals for top sectors."""
    profile = UserProfile(**state["profile"])

    sectors = ["tech", "healthcare", "energy", "financials"]
    if (profile.sector_preference
            and profile.sector_preference.value == "specific_industry_interest"
            and profile.sector_detail):
        sectors = [profile.sector_detail.lower()]

    # Pass 1 — sector-level signals
    sentiment    = SECTOR_TOOLS[0].invoke({"sectors": sectors})   # Signal A
    fundamentals = SECTOR_TOOLS[1].invoke({"sectors": sectors})   # Signal B
    macro        = SECTOR_TOOLS[2].invoke({})                     # Signal C
    momentum     = SECTOR_TOOLS[3].invoke({"sectors": sectors})   # Signal D

    # Pre-score to find top 3 sectors (LLM does full scoring in synthesize_node)
    sector_scores = {s: _compute_sector_score(s, sentiment, fundamentals, macro, momentum) for s in sectors}
    top_sectors = sorted(sector_scores, key=lambda x: sector_scores[x], reverse=True)[:3]

    # Pass 2 — stock-level signals for top sectors only
    top_tickers = [
        stock["ticker"]
        for sector in top_sectors
        for stock in SECTOR_STOCKS.get(sector, [])
    ]

    stock_sentiment    = STOCK_TOOLS[0].invoke({"tickers": top_tickers})  # Signal E
    stock_fundamentals = STOCK_TOOLS[1].invoke({"tickers": top_tickers})  # Signal F
    stock_momentum     = STOCK_TOOLS[2].invoke({"tickers": top_tickers})  # Signal G

    signals = {
        # Sector-level
        "signal_a_sentiment":    sentiment,
        "signal_b_fundamentals": fundamentals,
        "signal_c_macro":        macro,
        "signal_d_momentum":     momentum,
        # Pre-scored ranking (for reference)
        "top_sectors":           top_sectors,
        "sector_stocks":         {s: SECTOR_STOCKS.get(s, []) for s in top_sectors},
        # Stock-level
        "signal_e_stock_sentiment":    stock_sentiment,
        "signal_f_stock_fundamentals": stock_fundamentals,
        "signal_g_stock_momentum":     stock_momentum,
    }

    return {
        "signals": signals,
        "current_phase": "synthesize",
    }


def synthesize_node(state: AgentState) -> dict:
    """
    The big one: take the user profile + all four signals and generate
    a personalized investment recommendation.

    This is where the LLM acts as a synthesizer/tiebreaker.
    """
    llm = get_llm(temperature=0.4)  # slightly more creative for recommendations
    profile = UserProfile(**state["profile"])
    signals = state["signals"]

    synthesis_prompt = f"""You are scoring investment options to find the best fit for this specific user.

USER PROFILE:
{json.dumps(profile.model_dump(), indent=2, default=str)}

═══════════════════════════════════════════════
PASS 1 — SECTOR-LEVEL SIGNALS
═══════════════════════════════════════════════

SIGNAL A — NEWS MOOD (sector sentiment, -1.0 to 1.0):
{json.dumps(signals.get('signal_a_sentiment', {}), indent=2)}

SIGNAL B — COMPANY HEALTH (sector fundamentals, health_score -1.0 to 1.0):
{json.dumps(signals.get('signal_b_fundamentals', {}), indent=2)}

SIGNAL C — ECONOMY VIBE (macro, applies to all sectors equally):
{json.dumps(signals.get('signal_c_macro', {}), indent=2)}

SIGNAL D — PRICE TREND (sector ETF price action — supporting signal only):
{json.dumps(signals.get('signal_d_momentum', {}), indent=2)}

Pre-scored top sectors (use as a starting point, apply full scoring below):
{json.dumps(signals.get('top_sectors', []), indent=2)}

═══════════════════════════════════════════════
PASS 2 — STOCK-LEVEL SIGNALS (top sectors only)
═══════════════════════════════════════════════

STOCKS IN TOP SECTORS:
{json.dumps(signals.get('sector_stocks', {}), indent=2)}

SIGNAL E — NEWS MOOD per stock (sentiment_score -1.0 to 1.0):
{json.dumps(signals.get('signal_e_stock_sentiment', {}), indent=2)}

SIGNAL F — COMPANY HEALTH per stock (health_score -1.0 to 1.0, volatility label):
{json.dumps(signals.get('signal_f_stock_fundamentals', {}), indent=2)}

SIGNAL G — PRICE TREND per stock (momentum_score -1.0 to 1.0, recent % changes):
{json.dumps(signals.get('signal_g_stock_momentum', {}), indent=2)}

═══════════════════════════════════════════════
SCORING INSTRUCTIONS
═══════════════════════════════════════════════

STEP 1 — SCORE AND GRADE EACH SECTOR

Composite score = (news_mood × 0.25) + (company_health × 0.35) + (economy_vibe × 0.25) + (price_trend × 0.15)

Note: price_trend (Signal D) is price action only — a supporting signal, not a primary driver.
Economy vibe (Signal C) applies equally to all sectors — use the macro_score value directly.

Adjust weights by time horizon:
  - Under 2 years:  price_trend → 0.25, company_health → 0.25
  - 2–7 years:      use defaults above
  - 7+ years:       company_health → 0.45, price_trend → 0.05

Apply profile filter after scoring:
  - conservative risk (preserve_capital or sell_everything): remove any sector where
    company_health score < 0 OR news_mood score < -0.2
  - moderate risk (steady_growth or hold): remove sectors where 3+ signals are below 0
  - aggressive risk (comfortable_with_swings or maximum_growth / buy_more): keep all, flag risks

Map composite score to grade:
  A = 0.6 to 1.0    → Strong
  B = 0.2 to 0.59   → Moderate
  C = -0.19 to 0.19 → Neutral
  D = below -0.2    → Weak

Select top 3 sectors by composite score. Only A and B grades move forward unless fewer exist.

STEP 2 — SCORE STOCKS WITHIN TOP SECTORS

Composite score = (news_mood × 0.25) + (company_health × 0.35) + (economy_vibe × 0.25) + (price_trend × 0.15)
Use the same macro_score from Signal C for economy_vibe on all stocks.

Additional filter: does the stock's volatility match the user's drawdown tolerance?
  - preserve_capital or sell_everything: exclude "high" volatility stocks
  - steady_growth or hold: flag "high" volatility but keep
  - comfortable_with_swings or buy_more: keep all

Select top 3 stocks per sector by composite score.

═══════════════════════════════════════════════
OUTPUT FORMAT — follow exactly
═══════════════════════════════════════════════

CATEGORIES:

[Sector Name] | Grade: [A/B/C/D] | Confidence: [Strong/Moderate/Cautious]
Why: [one plain-English sentence tied to this specific user's profile]
Driven by: [which signals scored highest — plain English]
Watch out: [any conflicting signal or honest caveat — one sentence]

(repeat for top 3 sectors)

---

STOCKS IN [Sector Name]:

[TICKER] — [Full Company Name] | Grade: [A/B/C/D] | Mood: [Bullish/Neutral/Stable/Cautious]
Why it fits you: [one sentence referencing user's specific risk/goal/horizon]
Risk: [one plain-English risk flag]

(repeat for top 3 stocks per sector, then move to next sector)

---

SIGNAL SUMMARY:
News Mood: [one plain-English line]
Company Health: [one plain-English line]
Economy Vibe: [one plain-English line]
Price Trend: [one plain-English line — remind user this is recent price action, not a forecast]
Overall: [do all signals agree, or are there conflicts? Be honest in one sentence.]

---

NEXT STEP:
[One concrete action this specific user can take today. Name a real platform or resource.]

═══════════════════════════════════════════════
LANGUAGE RULES — always apply
═══════════════════════════════════════════════
- Match language to experience level: {profile.experience_level}
  - beginner: no tickers, explain every term, keep it very simple
  - intermediate: tickers are fine, light explanations
  - advanced: be direct, skip the hand-holding
- Never show raw scores or numbers to the user
- Never say "based on your profile" more than once
- Never use jargon without immediately explaining it in the same sentence"""

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT)]
        + state["messages"]
        + [HumanMessage(content=synthesis_prompt)]
    )

    return {
        "messages": [AIMessage(content=_text(response))],
        "current_phase": "done",
    }


# ── Graph Construction ─────────────────────────────────────────────

def route_by_phase(state: AgentState) -> str:
    """Conditional edge: route to the next node based on current_phase."""
    return state["current_phase"]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("greet", greet_node)
    graph.add_node("experience", experience_node)
    graph.add_node("risk", risk_node)
    graph.add_node("goals", goals_node)
    graph.add_node("constraints", constraints_node)
    graph.add_node("motivation", motivation_node)
    graph.add_node("confirm", confirm_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("synthesize", synthesize_node)

    # Route from START to the correct node based on current_phase.
    # This replaces set_entry_point so each graph.invoke() lands on the right node.
    graph.add_conditional_edges(
        START,
        lambda state: state["current_phase"],
        {
            "greet":       "greet",
            "experience":  "experience",
            "risk":        "risk",
            "goals":       "goals",
            "constraints": "constraints",
            "motivation":  "motivation",
            "confirm":     "confirm",
            "analyze":     "analyze",
        },
    )

    graph.add_edge("greet", END)
    graph.add_edge("experience", END)
    graph.add_edge("risk", END)
    graph.add_edge("goals", END)
    graph.add_edge("constraints", END)
    graph.add_edge("motivation", END)
    graph.add_edge("confirm", END)
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()