"""
main.py — Terminal UI for the Agentic Finance Bro.

Run with: python main.py

This is the MVP interface. It:
1. Starts the LangGraph agent
2. Prints the greeting
3. Loops: get user input → run the appropriate node → print response
4. After confirmation, runs analysis + synthesis automatically
5. Exports the final profile JSON to user_profile.json
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

# Terminal colors (works on most terminals)

class Colors:
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _render_md(text: str) -> str:
    """Convert basic markdown to terminal formatting."""
    # **bold** → terminal bold
    text = re.sub(r"\*\*(.+?)\*\*", rf"{Colors.BOLD}\1{Colors.RESET}", text)
    # Markdown list bullets (* item or *   item) → • item
    text = re.sub(r"^\*\s+", "  • ", text, flags=re.MULTILINE)
    return text


def print_agent(text: str):
    """Print agent message in blue with markdown rendered."""
    rendered = _render_md(text)
    print(f"\n{Colors.BLUE}{Colors.BOLD}Finance Bro:{Colors.RESET} {rendered}")


def print_system(text: str):
    """Print system info in dim."""
    print(f"{Colors.DIM}{text}{Colors.RESET}")


def print_profile_json(profile: UserProfile):
    """Pretty-print the profile JSON."""
    print(f"\n{Colors.CYAN}{'═' * 60}")
    print(f"  EXPORTED USER PROFILE (JSON)")
    print(f"{'═' * 60}{Colors.RESET}")
    profile_dict = profile.model_dump(mode="json")
    print(json.dumps(profile_dict, indent=2))
    print(f"{Colors.CYAN}{'═' * 60}{Colors.RESET}")


# Phase routing

# Maps current_phase → which node to invoke next after user input
PHASE_TO_NODE = {
    "experience": "experience",
    "risk": "risk",
    "goals": "goals",
    "constraints": "constraints",
    "motivation": "motivation",
    "confirm": "confirm",
}


# Main loop 

def main():
    # Check for API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print(f"\n{Colors.YELLOW}⚠  GOOGLE_API_KEY not set.{Colors.RESET}")
        print("Run: export GOOGLE_API_KEY=\"your-key\"")
        print("Or add it to your .env file.\n")
        sys.exit(1)

    print(f"\n{Colors.GREEN}{'═' * 60}")
    print(f"  AGENTIC FINANCE BRO")
    print(f"  Terminal MVP")
    print(f"{'═' * 60}{Colors.RESET}")
    print_system("Type 'quit' or 'exit' to stop at any time.")
    print_system("Type 'profile' to see the current profile state.\n")

    # Build the graph
    graph = build_graph()

    # Initialize state
    state = {
        "messages": [],
        "profile": UserProfile().model_dump(),
        "current_phase": "greet",
        "signals": {},
    }

    # Step 1: Run the greeting node 
    print_system("Starting agent...")
    result = graph.invoke(state, config={"run_name": "greet"})

    # Update state from result
    state["messages"] = result["messages"]
    state["current_phase"] = result["current_phase"]

    # Print the greeting
    last_ai_msg = [m for m in result["messages"] if hasattr(m, "content") and m.type == "ai"][-1]
    print_agent(last_ai_msg.content)

    # Step 2: Conversation loop 
    while True:
        # Get user input
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
            if status["missing"]:
                print_system(f"Missing fields: {', '.join(status['missing'])}")
            else:
                print_system("Profile is complete!")
            continue

        # Figure out which node to run next
        current_phase = state["current_phase"]
        next_node = PHASE_TO_NODE.get(current_phase)

        if current_phase == "done":
            print_agent("We're all done! Your recommendation is above. Type 'profile' to see your exported profile, or 'quit' to exit.")
            continue

        if not next_node:
            print_system(f"Unknown phase: {current_phase}. This shouldn't happen.")
            break

        # Add the user message to state
        state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]

        # Run the next node
        print_system(f"[Processing: {next_node} phase...]")

        try:
            result = graph.invoke(state, config={"run_name": next_node})
        except Exception as e:
            print(f"\n{Colors.YELLOW}Error: {e}{Colors.RESET}")
            print_system("Try again or check your API key.")
            continue

        # Update state
        state["messages"] = result["messages"]
        state["current_phase"] = result["current_phase"]
        if result.get("profile"):
            state["profile"] = result["profile"]

        # Print the agent's response
        ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
        if ai_messages:
            print_agent(ai_messages[-1].content)

        # If we just confirmed, auto-run analysis + synthesis
        if state["current_phase"] == "analyze":
            print_system("\n[Running market analysis — calling Signals A, B, C, D...]")

            # Run analyze → synthesize (these are chained, no user input needed)
            state["messages"] = list(state["messages"])
            result = graph.invoke(state, config={"run_name": "analyze"})

            state["messages"] = result["messages"]
            state["current_phase"] = result["current_phase"]
            state["signals"] = result.get("signals", {})
            state["profile"] = result.get("profile", state["profile"])

            # Print the recommendation
            ai_messages = [m for m in result["messages"] if hasattr(m, "type") and m.type == "ai"]
            if ai_messages:
                print_agent(ai_messages[-1].content)

            # Save profile to file (silently)
            profile = UserProfile(**state["profile"])
            output_path = "user_profile.json"
            with open(output_path, "w") as f:
                json.dump(profile.model_dump(mode="json"), f, indent=2)
            print_system(f"\nProfile saved to {output_path}. Type 'profile' to inspect it.")

            state["current_phase"] = "done"


if __name__ == "__main__":
    main()