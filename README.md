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
| `tools.py` | Seven signal tools (A–G); A and E use live FinBERT inference |
| `main.py` | Terminal UI loop |
| `test_agent.py` | Smoke tests and non-interactive end-to-end run |
| `finbert_adapter.py` | Bridge between `tools.py` and the local FinBERT model |
| `finbert_scorer/` | Fine-tuned FinBERT package (model weights tracked via git-lfs) |

### Signals

The agent runs two passes: sector-level first, then stock-level for the top 3 sectors.

**Pass 1 — Sector signals**

| Signal | Name | Status |
|--------|------|--------|
| A — Sentiment | Fine-tuned FinBERT on stub headlines (swap for NewsAPI) | **Live** |
| B — Fundamentals | P/E, debt-to-equity, revenue growth | Stub → SimFin / Alpha Vantage |
| C — Macro | Fed funds rate, CPI, unemployment | Stub → FRED API |
| D — Momentum | Sector ETF price action | Stub → yfinance |

**Pass 2 — Stock signals (top 3 sectors only)**

| Signal | Name | Status |
|--------|------|--------|
| E — Stock Sentiment | Fine-tuned FinBERT on stub headlines (swap for NewsAPI) | **Live** |
| F — Stock Fundamentals | Per-stock P/E, volatility, revenue growth | Stub → SimFin / Alpha Vantage |
| G — Stock Momentum | Per-stock price action | Stub → yfinance |

## Setup

### 1. Install git-lfs (once per machine)

The fine-tuned FinBERT model weights (~418 MB) are stored via git-lfs. You need git-lfs installed before cloning, otherwise the `.safetensors` files will download as small pointer stubs and the agent will fail to load the model.

**macOS:**
```bash
brew install git-lfs
git lfs install
```

**Linux:**
```bash
sudo apt install git-lfs   # or: sudo yum install git-lfs
git lfs install
```

**Windows:**
Download the installer from [git-lfs.com](https://git-lfs.com), then run:
```bash
git lfs install
```

### 2. Clone and install

```bash
git clone <repo-url>
cd finance-bro-agent
pip install -r requirements.txt
```

### 3. Set your API key

Edit `.env`:

```
GOOGLE_API_KEY=your-gemini-api-key-here
```

The app loads this automatically via `python-dotenv`. You can also export it in your shell:

```bash
export GOOGLE_API_KEY="your-key"
```

### 4. Run the agent

```bash
python main.py
```

**Tip:** type `profile` at any point during the conversation to inspect the current user profile JSON. Type `quit` or `exit` to stop.

### 5. Run the tests

```bash
python test_agent.py
```

This runs unit tests for the schema and tools (no API key needed), then optionally runs a full end-to-end agent pass with hardcoded answers if `GOOGLE_API_KEY` is set.

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

Signals A and E run live FinBERT inference but are fed stub headlines. All other signals return hardcoded values. To wire in real data:

- `get_sentiment_signal` / `get_stock_sentiment_signal` — replace `_SECTOR_HEADLINES` / `_STOCK_HEADLINES` in `tools.py` with live NewsAPI queries
- `get_fundamentals_signal` — call SimFin or Alpha Vantage sector endpoints
- `get_macro_signal` — call FRED (`fred.stlouisfed.org/docs/api/fred/`)
- `get_momentum_signal` / `get_stock_momentum_signal` — use `yfinance` to pull ETF and stock price history

## Requirements

- Python 3.10+
- Google Gemini API key
