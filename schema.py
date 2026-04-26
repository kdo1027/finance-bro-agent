"""
schema.py — Pydantic models for the user investment profile
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Enums for structured intake fields

class ExperienceLevel(str, Enum):
    BEGINNER = "beginner"           # 1-2 on the scale
    INTERMEDIATE = "intermediate"   # 3
    ADVANCED = "advanced"           # 4-5


class InvestmentType(str, Enum):
    STOCKS = "individual_stocks"
    ETFS = "etfs_or_mutual_funds"
    BONDS = "bonds"
    CRYPTO = "crypto"
    OPTIONS = "options_or_futures"
    NONE = "none_yet"


class DrawdownResponse(str, Enum):
    """Q3: What would you do if portfolio dropped 25%?"""
    BUY_MORE = "buy_more"
    HOLD = "hold"
    SELL_SOME = "sell_some"
    SELL_ALL = "sell_everything"


class RiskStatement(str, Enum):
    """Q4: Which statement fits you best?"""
    PRESERVE = "preserve_capital"
    STEADY = "steady_growth"
    COMFORTABLE = "comfortable_with_swings"
    MAXIMUM = "maximum_growth"


class InvestingGoal(str, Enum):
    RETIREMENT = "retirement"
    HOUSE = "house_down_payment"
    WEALTH = "general_wealth_building"
    INCOME = "short_term_income"
    OTHER = "other"


class TimeHorizon(str, Enum):
    LESS_THAN_1 = "less_than_1_year"
    ONE_TO_THREE = "1_to_3_years"
    THREE_TO_TEN = "3_to_10_years"
    TEN_PLUS = "10_plus_years"


class TargetReturn(str, Enum):
    UNDER_5 = "under_5_percent"
    FIVE_TO_TEN = "5_to_10_percent"
    TEN_TO_FIFTEEN = "10_to_15_percent"
    FIFTEEN_PLUS = "15_plus_percent"


class SectorPreference(str, Enum):
    ESG = "esg_sustainability"
    EXCLUDE_TOBACCO_DEFENSE = "exclude_tobacco_defense"
    SPECIFIC_INDUSTRY = "specific_industry_interest"
    NO_PREFERENCE = "no_preference"


class PortfolioPercent(str, Enum):
    """Q9: How much of total savings does this represent?"""
    UNDER_10 = "under_10_percent"
    TEN_TO_25 = "10_to_25_percent"
    TWENTY_FIVE_TO_50 = "25_to_50_percent"
    FIFTY_PLUS = "50_plus_percent"


class SignalGrade(str, Enum):
    """Categorical grade mapped from a composite signal score (-1.0 to 1.0)."""
    A = "A"  # Strong   ( 0.6 to  1.0)
    B = "B"  # Moderate ( 0.2 to  0.59)
    C = "C"  # Neutral  (-0.19 to 0.19)
    D = "D"  # Weak     (-0.2  to -1.0)

    @classmethod
    def from_score(cls, score: float) -> "SignalGrade":
        if score >= 0.6:
            return cls.A
        if score >= 0.2:
            return cls.B
        if score >= -0.19:
            return cls.C
        return cls.D


# Main user profile

class UserProfile(BaseModel):
    """
    Complete user investment profile built from intake questions.
    This object is passed to the LLM synthesizer alongside Signals A-D.
    """

    # Investment knowledge (Q1-Q2)
    experience_level: Optional[ExperienceLevel] = Field(
        None, description="Self-rated investing experience (Q1)"
    )
    prior_investments: Optional[list[InvestmentType]] = Field(
        None, description="Which asset classes the user has invested in (Q2)"
    )

    # Risk tolerance (Q3-Q4)
    drawdown_response: Optional[DrawdownResponse] = Field(
        None, description="Behavior if portfolio dropped 25% (Q3)"
    )
    risk_statement: Optional[RiskStatement] = Field(
        None, description="Best-fit risk tolerance statement (Q4)"
    )

    # Goals and horizon (Q5-Q7)
    investing_goal: Optional[InvestingGoal] = Field(
        None, description="Primary investing goal (Q5)"
    )
    time_horizon: Optional[TimeHorizon] = Field(
        None, description="When the user expects to need the money (Q6)"
    )
    target_return: Optional[TargetReturn] = Field(
        None, description="Target annualized return (Q7)"
    )

    # Constraints and preferences (Q8-Q9)
    sector_preference: Optional[SectorPreference] = Field(
        None, description="Sector preferences or exclusions (Q8)"
    )
    sector_detail: Optional[str] = Field(
        None, description="Free text detail if they specified industries (Q8)"
    )
    portfolio_percent: Optional[PortfolioPercent] = Field(
        None, description="What % of total savings this represents (Q9)"
    )

    # Framing context (Q10)
    investing_motivation: Optional[str] = Field(
        None, description="One-sentence reason for investing now (Q10)"
    )

    def completion_status(self) -> dict:
        """Returns which fields are filled vs missing."""
        filled = {}
        missing = []
        for name, field_info in self.model_fields.items():
            value = getattr(self, name)
            if value is not None:
                filled[name] = value
            else:
                missing.append(name)
        return {"filled": filled, "missing": missing}

    def is_complete(self) -> bool:
        """Check if all required fields are populated."""
        status = self.completion_status()
        # sector_detail is optional (only needed if sector_preference is specific_industry)
        required_missing = [f for f in status["missing"] if f != "sector_detail"]
        return len(required_missing) == 0


# ── Extraction models (used by LLM to parse one answer at a time) ──

class ExtractedExperience(BaseModel):
    """LLM extracts these from the user's response to Q1-Q2."""
    experience_level: Optional[ExperienceLevel] = None
    prior_investments: Optional[list[InvestmentType]] = None


class ExtractedRisk(BaseModel):
    """LLM extracts these from the user's response to Q3-Q4."""
    drawdown_response: Optional[DrawdownResponse] = None
    risk_statement: Optional[RiskStatement] = None


class ExtractedGoals(BaseModel):
    """LLM extracts these from the user's response to Q5-Q7."""
    investing_goal: Optional[InvestingGoal] = None
    time_horizon: Optional[TimeHorizon] = None
    target_return: Optional[TargetReturn] = None


class ExtractedConstraints(BaseModel):
    """LLM extracts these from the user's response to Q8-Q9."""
    sector_preference: Optional[SectorPreference] = None
    sector_detail: Optional[str] = None
    portfolio_percent: Optional[PortfolioPercent] = None


class ExtractedMotivation(BaseModel):
    """LLM extracts this from the user's response to Q10."""
    investing_motivation: Optional[str] = None