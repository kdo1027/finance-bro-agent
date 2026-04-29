"""
main.py — Terminal UI for the Agentic Finance Bro.

Run with: python main.py

Structured intake questions (Q1-Q9) are shown as numbered menus — no typing
required for those. Only Q10 (motivation) is free text. This bypasses LLM
extraction mismatches and saves API calls on the intake flow.
"""

import json
import os
import re
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from agent import build_graph
from schema import UserProfile


# ── Terminal colors ────────────────────────────────────────────────────────────
class Colors:
    BLUE   = "\033[94m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    DIM    = "\033[2m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def _render_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", rf"{Colors.BOLD}\1{Colors.RESET}", text)
    text = re.sub(r"^\*\s+", "  • ", text, flags=re.MULTILINE)
    return text


def print_agent(text: str):
    rendered = _render_md(text)
    print(f"\n{Colors.BLUE}{Colors.BOLD}Finance Bro:{Colors.RESET} {rendered}")


def print_system(text: str):
    print(f"{Colors.DIM}{text}{Colors.RESET}")


def print_profile_json(profile: UserProfile):
    print(f"\n{Colors.CYAN}{'═' * 60}")
    print(f"  EXPORTED USER PROFILE (JSON)")
    print(f"{'═' * 60}{Colors.RESET}")
    print(json.dumps(profile.model_dump(mode="json"), indent=2))
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")


# ── Numbered menu definitions for structured phases ────────────────────────────
# Each menu has: question text, list of (display label, enum value), multi-select flag
PHASE_MENUS = {

    "experience": [
        {
            "question": "Q1 — How would you rate your investing experience?",
            "options": [
                ("1  Just starting out — never invested before",             "beginner"),
                ("2  Tried it a bit — bought stocks or ETFs once or twice",  "beginner"),
                ("3  Some experience — invested regularly for 1-3 years",    "intermediate"),
                ("4  Pretty experienced — active investor, 3+ years",        "advanced"),
                ("5  Very experienced — options, futures, complex strategies","advanced"),
            ],
            "key":   "experience_level",
            "multi": False,
        },
        {
            "question": "Q2 — What have you invested in before? (enter numbers separated by commas, e.g.  1,3)",
            "options": [
                ("1  Individual stocks",     "individual_stocks"),
                ("2  ETFs or mutual funds",  "etfs_or_mutual_funds"),
                ("3  Bonds",                 "bonds"),
                ("4  Crypto",                "crypto"),
                ("5  Options or futures",    "options_or_futures"),
                ("6  Nothing yet",           "none_yet"),
            ],
            "key":   "prior_investments",
            "multi": True,
        },
    ],

    "risk": [
        {
            "question": "Q3 — If your portfolio dropped 25% in one month, what would you do?",
            "options": [
                ("1  Buy more — great buying opportunity",  "buy_more"),
                ("2  Hold — stay the course",               "hold"),
                ("3  Sell some — reduce my exposure",       "sell_some"),
                ("4  Sell everything — get out now",        "sell_everything"),
            ],
            "key":   "drawdown_response",
            "multi": False,
        },
        {
            "question": "Q4 — Which statement fits you best?",
            "options": [
                ("1  I want to preserve what I have, even if returns are modest", "preserve_capital"),
                ("2  I want steady growth with limited downside",                  "steady_growth"),
                ("3  I'm comfortable with swings for better long-term returns",    "comfortable_with_swings"),
                ("4  Maximum growth — I can handle big drops",                     "maximum_growth"),
            ],
            "key":   "risk_statement",
            "multi": False,
        },
    ],

    "goals": [
        {
            "question": "Q5 — What's your main reason for investing?",
            "options": [
                ("1  Retirement",              "retirement"),
                ("2  House down payment",      "house_down_payment"),
                ("3  General wealth building", "general_wealth_building"),
                ("4  Short-term income",       "short_term_income"),
                ("5  Other",                   "other"),
            ],
            "key":   "investing_goal",
            "multi": False,
        },
        {
            "question": "Q6 — When do you expect to need this money?",
            "options": [
                ("1  Less than 1 year", "less_than_1_year"),
                ("2  1 to 3 years",     "1_to_3_years"),
                ("3  3 to 10 years",    "3_to_10_years"),
                ("4  10+ years",        "10_plus_years"),
            ],
            "key":   "time_horizon",
            "multi": False,
        },
        {
            "question": "Q7 — What's your target annual return?",
            "options": [
                ("1  Under 5%  — keep it safe",    "under_5_percent"),
                ("2  5 to 10%  — steady growth",   "5_to_10_percent"),
                ("3  10 to 15% — ambitious",        "10_to_15_percent"),
                ("4  15%+      — aggressive",       "15_plus_percent"),
            ],
            "key":   "target_return",
            "multi": False,
        },
    ],

    "constraints": [
        {
            "question": "Q8 — Any sector preferences or things to avoid?",
            "options": [
                ("1  ESG / sustainability focus",          "esg_sustainability"),
                ("2  Exclude tobacco and defense",         "exclude_tobacco_defense"),
                ("3  I have a specific industry in mind",  "specific_industry_interest"),
                ("4  No preference",                       "no_preference"),
            ],
            "key":   "sector_preference",
            "multi": False,
        },
        {
            "question": "Q9 — What percentage of your total savings is this investment?",
            "options": [
                ("1  Under 10%",    "under_10_percent"),
                ("2  10 to 25%",    "10_to_25_percent"),
                ("3  25 to 50%",    "25_to_50_percent"),
                ("4  50% or more",  "50_plus_percent"),
            ],
            "key":   "portfolio_percent",
            "multi": False,
        },
    ],
}


def _prompt_number(prompt: str, max_val: int) -> int | None:
    """Show a prompt and return a valid 1-based int, or None on interrupt."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.isdigit() and 1 <= int(raw) <= max_val:
            return int(raw)
        print(f"  {Colors.YELLOW}Please enter a number between 1 and {max_val}.{Colors.RESET}")


def show_menu(phase: str) -> str | None:
    """Display numbered menus for a structured phase.

    Returns a clearly formatted answer string the extraction LLM can parse
    reliably, or None if the user interrupted.
    For 'specific_industry_interest' it asks a follow-up free-text question.
    """
    menus = PHASE_MENUS.get(phase)
    if not menus:
        return None

    answers: dict[str, object] = {}

    for menu in menus:
        print(f"\n{Colors.CYAN}{menu['question']}{Colors.RESET}")
        for label, _ in menu["options"]:
            print(f"  {Colors.DIM}{label}{Colors.RESET}")

        if menu["multi"]:
            while True:
                try:
                    raw = input(f"\n{Colors.GREEN}Your choices:{Colors.RESET} ").strip()
                except (EOFError, KeyboardInterrupt):
                    return None
                parts   = [p.strip() for p in raw.split(",")]
                valid   = all(p.isdigit() and 1 <= int(p) <= len(menu["options"]) for p in parts)
                if valid:
                    values = [menu["options"][int(p) - 1][1] for p in parts]
                    answers[menu["key"]] = values
                    break
                print(f"  {Colors.YELLOW}Enter numbers separated by commas (e.g. 1,3).{Colors.RESET}")
        else:
            choice = _prompt_number(
                f"\n{Colors.GREEN}Your choice (1-{len(menu['options'])}):{Colors.RESET} ",
                len(menu["options"]),
            )
            if choice is None:
                return None
            answers[menu["key"]] = menu["options"][choice - 1][1]

    # Follow-up: if user picked "specific industry", ask which one
    if answers.get("sector_preference") == "specific_industry_interest":
        try:
            detail = input(f"\n{Colors.GREEN}Which industry are you interested in?{Colors.RESET} ").strip()
            if detail:
                answers["sector_detail"] = detail
        except (EOFError, KeyboardInterrupt):
            return None

    # Build a clear, unambiguous string for the extraction LLM
    parts = []
    for key, val in answers.items():
        if isinstance(val, list):
            parts.append(f"{key}: {', '.join(val)}")
        else:
            parts.append(f"{key}: {val}")
    return ". ".join(parts)


# ── Phase routing ──────────────────────────────────────────────────────────────
PHASE_TO_NODE = {
    "experience":  "experience",
    "risk":        "risk",
    "goals":       "goals",
    "constraints": "constraints",
    "motivation":  "motivation",
    "confirm":     "confirm",
}


# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        print(f"\n{Colors.YELLOW}⚠  GOOGLE_API_KEY not set.{Colors.RESET}")
        print("Run: export GOOGLE_API_KEY=\"your-key\"")
        print("Or add it to your .env file.\n")
        sys.exit(1)

    print(f"\n{Colors.GREEN}{'═' * 60}")
    print(f"  FINANCE BRO")
    print(f"  Your Personal Investment Guide")
    print(f"{'═' * 60}{Colors.RESET}")
    print_system("Structured questions will show numbered options — just pick a number.")
    print_system("Type 'profile' anytime to see your answers. Type 'quit' to exit.\n")

    graph = build_graph()
    state = {
        "messages":      [],
        "profile":       UserProfile().model_dump(),
        "current_phase": "greet",
        "signals":       {},
    }

    # Greeting
    print_system("Starting...")
    result = graph.invoke(state, config={"run_name": "greet"})
    state["messages"]      = result["messages"]
    state["current_phase"] = result["current_phase"]

    last_ai = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"][-1]
    print_agent(last_ai.content)

    # Conversation loop
    while True:
        current_phase = state["current_phase"]

        if current_phase == "done":
            print_agent("All done! Your recommendation is above. Type 'profile' to review your answers, or 'quit' to exit.")
            try:
                cmd = input(f"\n{Colors.GREEN}You:{Colors.RESET} ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd in ("quit", "exit"):
                break
            if cmd == "profile":
                print_profile_json(UserProfile(**state["profile"]))
            continue

        # ── Get user input ────────────────────────────────────────────────────
        if current_phase in PHASE_MENUS:
            # Structured question — show numbered menu
            user_input = show_menu(current_phase)
            if user_input is None:
                print("\n\nGoodbye!")
                break
        else:
            # Free text — motivation (Q10) and confirm
            try:
                user_input = input(f"\n{Colors.GREEN}You:{Colors.RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "profile":
            profile = UserProfile(**state["profile"])
            print_profile_json(profile)
            status = profile.completion_status()
            print_system(f"Missing: {', '.join(status['missing'])}" if status["missing"] else "Profile complete!")
            continue

        next_node = PHASE_TO_NODE.get(current_phase)
        if not next_node:
            print_system(f"Unknown phase: {current_phase}.")
            break

        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]
        print_system(f"[{next_node}...]")

        try:
            result = graph.invoke(state, config={"run_name": next_node})
        except Exception as e:
            print(f"\n{Colors.YELLOW}Error: {e}{Colors.RESET}")
            print_system("Try again or check your API key.")
            continue

        state["messages"]      = result["messages"]
        state["current_phase"] = result["current_phase"]
        if result.get("profile"):
            state["profile"] = result["profile"]

        ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_messages:
            print_agent(ai_messages[-1].content)

        # Auto-run analysis + synthesis after confirmation
        if state["current_phase"] == "analyze":
            print_system("\n[Analyzing market — sector signals (A-D), then stock signals (E-G) for top sectors...]")

            result = graph.invoke(state, config={"run_name": "analyze"})
            state["messages"]      = result["messages"]
            state["current_phase"] = result["current_phase"]
            state["signals"]       = result.get("signals", {})
            state["profile"]       = result.get("profile", state["profile"])

            ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
            if ai_messages:
                print_agent(ai_messages[-1].content)

            profile = UserProfile(**state["profile"])
            with open("user_profile.json", "w") as f:
                json.dump(profile.model_dump(mode="json"), f, indent=2)
            print_system("\nProfile saved to user_profile.json. Type 'profile' to inspect it.")

            state["current_phase"] = "done"


if __name__ == "__main__":
    main()
