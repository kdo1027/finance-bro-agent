# Agentic Finance Bro

A conversational retail investment advisor built with LangGraph. The agent walks a user through a 10-question intake interview, then synthesizes four independent market signals into a personalized sector allocation recommendation.

## Architecture

```
greet → experience → risk → goals → constraints → motivation → confirm → analyze → synthesize → done
```

Each node in the graph handles one phase of the conversation:

| File | Purpose |
|------|---------|
| `agent.py` | LangGraph state machine — all nodes and graph wiring |
| `schema.py` | Pydantic models for the user profile and LLM extraction |
| `tools.py` | Four signal tools (stub data — replace with live APIs) |
| `main.py` | Terminal UI loop |
| `test_agent.py` | Smoke tests and non-interactive end-to-end run |

### Signals

| Signal | Source (stub → real) |
|--------|----------------------|
| A — Sentiment | FinBERT on news headlines (NewsAPI) |
| B — Fundamentals | P/E, debt-to-equity, revenue growth (SimFin / Alpha Vantage) |
| C — Macro | Fed funds rate, CPI, unemployment (FRED API) |
| D — Momentum | 30-day vs 90-day moving averages on sector ETFs (yfinance) |

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd finance-bro-agent
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
cp .env .env          # .env already exists — just fill it in
```

Edit `.env`:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

The app loads this automatically via `python-dotenv`. You can also export it in your shell:

```bash
export GOOGLE_API_KEY="your-key"
```

### 3. Run the agent

```bash
python main.py
```

**Tip:** type `profile` at any point during the conversation to inspect the current user profile JSON. Type `quit` or `exit` to stop.

### 4. Run the tests

```bash
python test_agent.py
```

This runs unit tests for the schema and tools (no API key needed), then optionally runs a full end-to-end agent pass with hardcoded answers if `OPENAI_API_KEY` is set.

## Usage example

```
Finance Bro: Hey! I'm your personal investment advisor...
             How would you rate your investing experience on a scale of 1-5?

You: I'd say a 3. I've bought some ETFs and individual stocks but never touched options.

[Processing: experience phase...]

Finance Bro: Nice — solid intermediate experience. Now a couple of risk questions...
```

After you confirm your profile, the agent automatically:
1. Calls all four signal tools
2. Synthesizes a recommendation referencing specific signal values
3. Exports your profile to `user_profile.json`

## Replacing stub data

The tools in `tools.py` return hardcoded values. To wire in real data, replace each stub function body:

- `get_sentiment_signal` — call NewsAPI, run headlines through a FinBERT pipeline
- `get_fundamentals_signal` — call SimFin or Alpha Vantage sector endpoints
- `get_macro_signal` — call FRED (`fred.stlouisfed.org/docs/api/fred/`)
- `get_momentum_signal` — use `yfinance` to pull ETF price history and compute MAs

## Requirements

- Python 3.10+
- Google Gemini API key
