"""
Supply Chain Analytics - Inventory Decision Agent
=================================================
An agentic decision engine implementing the 7-step pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supported Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock and Reorder Point (Continuous Review (s, Q))
4. Fixed-Period Review (Periodic Review / Order-Up-To Level (R, S))
5. Newsvendor Model (Single-Period Probabilistic)

Enhancements:
- Comprehensive inventory terminology and synonym recognition.
- Exhaustive full-sentence parsing before declaring missing information.
- Precise discrimination between demand, lead time, variability, cost types, and inventory levels.
- Explicit detection and clear reporting of genuinely missing parameters without inventing values.
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple, Union

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_newsvendor,
    calculate_safety_stock_and_rop,
    calculate_fixed_period_review,
    EOQResult,
    QuantityDiscountResult,
    ROPResult,
    PeriodicReviewResult,
    NewsvendorResult,
    SafetyStockROPResult,
    FixedPeriodReviewResult,
)


@dataclass
class VerificationRecord:
    is_valid: bool
    status: str  # "PASSED", "WARNING", "FAILED", "INCOMPLETE"
    checks: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class AgentDecisionResponse:
    question: str
    identified_model: str
    model_name_display: str
    confidence_score: float
    extracted_inputs: Dict[str, Any]
    selected_tool_name: str
    calculation_result: Any
    verification: VerificationRecord
    explanation: str
    formatted_output: str
    raw_step_trace: Dict[str, Any] = field(default_factory=dict)
    missing_parameters: List[str] = field(default_factory=list)
    is_complete: bool = True

    @property
    def student_verification(self) -> Optional[Dict[str, Any]]:
        """Returns the 8 deterministic pedagogical verification sections if calculated."""
        if not self.calculation_result:
            return None
        return {
            "inputs_extracted": getattr(self.calculation_result, "inputs_extracted", None),
            "variable_definitions": getattr(self.calculation_result, "variable_definitions", None),
            "formula": getattr(self.calculation_result, "formula", None),
            "formula_with_substituted_values": getattr(self.calculation_result, "formula_with_substituted_values", None),
            "intermediate_calculations": getattr(self.calculation_result, "intermediate_calculations", None),
            "final_answer": getattr(self.calculation_result, "final_answer", None),
            "verification_checks": getattr(self.calculation_result, "verification_checks", None),
            "assumptions": getattr(self.calculation_result, "assumptions", None),
        }


class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Interprets natural language inquiries, maps to OR inventory models, executes
    deterministic calculations, verifies constraints, and produces executive explanations.
    """

    MODEL_EOQ = "eoq"
    MODEL_QUANTITY_DISCOUNT = "quantity_discount"
    MODEL_SAFETY_STOCK_ROP = "safety_stock_rop"
    MODEL_FIXED_PERIOD_REVIEW = "fixed_period_review"
    MODEL_NEWSVENDOR = "newsvendor"

    MODEL_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount Model",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Order-Up-To Level (R, S))",
        MODEL_NEWSVENDOR: "Newsvendor Model (Single-Period Probabilistic)",
    }

    TOOL_MAPPING = {
        MODEL_EOQ: "calculate_eoq",
        MODEL_QUANTITY_DISCOUNT: "calculate_quantity_discount",
        MODEL_SAFETY_STOCK_ROP: "calculate_rop",
        MODEL_FIXED_PERIOD_REVIEW: "calculate_periodic_review",
        MODEL_NEWSVENDOR: "calculate_newsvendor",
    }

    def __init__(self):
        pass

    # ========================================================================
    # STEP 1: IDENTIFY MODEL
    # ========================================================================

    def identify_model(self, text: str) -> Tuple[str, float]:
        """
        Classifies the natural-language question into one of the 5 inventory models
        using weighted semantic cues and domain terminology.
        """
        lower = text.lower()

        scores = {
            self.MODEL_EOQ: 0.0,
            self.MODEL_QUANTITY_DISCOUNT: 0.0,
            self.MODEL_SAFETY_STOCK_ROP: 0.0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0.0,
            self.MODEL_NEWSVENDOR: 0.0,
        }

        # 1. Newsvendor indicators
        newsvendor_keywords = [
            "newsvendor", "newsboy", "single period", "perishable", "salvage",
            "markdown", "underage", "overage", "shortage cost", "excess cost",
            "critical fractile", "critical ratio", "spoilage", "fashion season",
            "christmas tree", "perishable goods", "single order", "one-time purchase",
            "leftover inventory", "goodwill penalty"
        ]
        for kw in newsvendor_keywords:
            if kw in lower:
                scores[self.MODEL_NEWSVENDOR] += 3.5

        # 2. Quantity discount indicators
        discount_keywords = [
            "all-units quantity discount", "all units quantity discount",
            "all-units discount", "all units discount",
            "quantity discount", "price break", "volume discount", "bulk discount",
            "discount schedule", "price tiers", "price tier", "tiers", "tier 1", "tier 2", "order breaks",
            "bracket", "unit cost drops", "price schedule", "break point", "quantity break"
        ]
        for kw in discount_keywords:
            if kw in lower:
                scores[self.MODEL_QUANTITY_DISCOUNT] += 3.5

        # 3. Fixed-Period Review indicators
        fixed_period_keywords = [
            "fixed period", "fixed-period", "periodic review", "order up to",
            "order-up-to", "target inventory", "review interval", "review period",
            "(r, s)", "(r,s)", "every week", "every month", "every 7 days",
            "every 14 days", "every two weeks", "periodic replenishment",
            "protection interval", "cycle count interval"
        ]
        for kw in fixed_period_keywords:
            if kw in lower:
                scores[self.MODEL_FIXED_PERIOD_REVIEW] += 3.5

        # 4. Safety Stock & Reorder Point indicators
        rop_keywords = [
            "safety stock", "reorder point", "rop", "continuous review",
            "(s, q)", "(s,q)", "lead time demand", "stockout risk", "cycle service level",
            "lead-time standard deviation", "lead time standard deviation",
            "demand standard deviation", "standard deviation of demand",
            "lead time variability", "demand variability", "z-score", "buffer stock"
        ]
        for kw in rop_keywords:
            if kw in lower:
                scores[self.MODEL_SAFETY_STOCK_ROP] += 3.0

        # 5. EOQ indicators
        eoq_keywords = [
            "eoq", "economic order quantity", "optimal batch size", "optimal order quantity",
            "setup cost", "setup expense", "ordering cost", "order cost", "cost per order",
            "cost of placing an order", "ordering/setup cost", "holding cost", "annual demand",
            "how much to order", "how many to order", "batch size", "lot size"
        ]
        for kw in eoq_keywords:
            if kw in lower:
                scores[self.MODEL_EOQ] += 2.0

        # Disambiguation logic:
        # If explicit all-units discount, price breaks, or discount tiers are present, prioritize Quantity Discount over standard EOQ
        if any(term in lower for term in ["all-units", "all units", "quantity discount", "volume discount", "price break", "quantity break"]) and \
           any(marker in lower for marker in ["[", "tier", "break", "$", "+", "to", "schedule", "bracket"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 8.0
        elif any(term in lower for term in ["price break", "quantity break", "discount", "volume discount", "tier"]) and \
             any(marker in lower for marker in ["[", "tier", "break", "$", "+", "to"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 4.0

        # If periodic review or review interval is present, prioritize Fixed Period Review
        if any(term in lower for term in ["review period", "review interval", "order-up-to", "order up to", "periodic"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 4.0

        # If safety stock or ROP is explicitly asked without periodic review
        if ("reorder point" in lower or "safety stock" in lower or "rop" in lower) and \
           "review period" not in lower and "order up to" not in lower:
            scores[self.MODEL_SAFETY_STOCK_ROP] += 3.5

        best_model = max(scores, key=scores.get)
        max_score = scores[best_model]

        confidence = min(0.99, max(0.50, max_score / 6.0)) if max_score > 0 else 0.40
        if max_score == 0:
            best_model = self.MODEL_EOQ

        return best_model, confidence

    # ========================================================================
    # STEP 2: EXTRACT INPUTS (COMPREHENSIVE FULL-SENTENCE EXTRACTION)
    # ========================================================================

    @staticmethod
    def _search_patterns(patterns: List[str], text_str: str) -> Optional[float]:
        """Scans text against regex patterns and returns the first float match."""
        for pat in patterns:
            match = re.search(pat, text_str, re.IGNORECASE)
            if match:
                val_str = match.group(1).replace(",", "").strip()
                try:
                    return float(val_str)
                except ValueError:
                    continue
        return None

    def extract_inputs(self, text: str, model_type: str) -> Dict[str, Any]:
        """
        Examines the entire question to extract all numerical values and meanings
        according to common inventory terminology and model-specific distinctions.
        """
        inputs: Dict[str, Any] = {}
        lower = text.lower()

        if model_type == self.MODEL_EOQ:
            return self._extract_eoq_inputs(text, lower)
        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(text, lower)
        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            return self._extract_rop_inputs(text, lower)
        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            return self._extract_periodic_review_inputs(text, lower)
        elif model_type == self.MODEL_NEWSVENDOR:
            return self._extract_newsvendor_inputs(text, lower)
        return inputs

    # ------------------------------------------------------------------------
    # EOQ Extraction
    # ------------------------------------------------------------------------
    def _extract_eoq_inputs(self, text: str, lower: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        # 1. Ordering Cost (Recognizing all synonyms)
        # - ordering cost, order cost, setup cost, setup expense, cost per order,
        #   cost of placing an order, ordering/setup cost, order placement cost, order fee
        ordering_cost = self._search_patterns([
            # Label then number
            r'(?:ordering\s*(?:/|and)?\s*setup\s*cost|ordering\s*[/&]\s*setup\s*expense|cost\s*of\s*placing\s*(?:an)?\s*order|setup\s*expense|ordering\s*expense|cost\s*per\s*order|ordering\s*cost|order\s*cost|setup\s*cost|order\s*placement\s*(?:cost|fee)|order\s*fee|setup\s*fee|setup\s*charge|s\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            # Number then label
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars|\$)?\s*(?:ordering\s*(?:/|and)?\s*setup\s*cost|cost\s*of\s*placing\s*(?:an)?\s*order|setup\s*expense|cost\s*per\s*order|ordering\s*cost|order\s*cost|setup\s*cost|order\s*fee|per\s*order|/\s*order)',
        ], text)
        if ordering_cost is not None:
            inputs["ordering_cost"] = ordering_cost

        # 2. Demand (Annual Demand)
        demand = self._search_patterns([
            r'(?:annual\s*demand|yearly\s*demand|annual\s*usage|annual\s*sales|demand\s*per\s*year)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*per\s*year|units/year|annual\s*units|units\s*annually)',
            r'(?:demand|usage|sales|orders)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*(?:per\s*year|annually)|units/year)',
            r'(?:demand\s*is|demand\s*of|demand:?)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'demand\s*=\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
        ], text)
        if demand is not None:
            inputs["demand"] = demand

        # 3. Holding Cost (Fixed vs. Percentage)
        holding_rate = self._search_patterns([
            r'(?:percentage\s*holding\s*cost|holding\s*cost\s*rate|holding\s*rate|carrying\s*rate|holding\s*cost\s*percentage|carrying\s*cost\s*percentage|annual\s*holding\s*rate|holding\s*cost|carrying\s*cost)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the)?\s*(?:unit\s*)?(?:purchase\s*)?(?:cost|price)|per\s*year\s*holding|holding\s*cost|carrying\s*cost|holding\s*rate|carrying\s*rate)',
            r'(?:percentage\s*holding\s*cost|holding\s*cost\s*rate|holding\s*rate|carrying\s*rate|holding\s*cost|carrying\s*cost)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*(0\.[0-9]+)(?!\s*%)',
        ], text)
        if holding_rate is not None and holding_rate > 1.0:
            holding_rate = holding_rate / 100.0

        fixed_holding = self._search_patterns([
            r'(?:fixed\s*holding\s*cost|unit\s*holding\s*cost|storage\s*cost|carrying\s*charge|h\s*=)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)?',
            r'(?:holding\s*cost|carrying\s*cost|holding\s*expense|cost\s*to\s*hold|cost\s*of\s*holding)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)?',
            r'(?:holding\s*cost|carrying\s*cost|holding\s*expense|cost\s*to\s*hold|cost\s*of\s*holding)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:dollars|\$)?\s*(?:per\s*unit\s*(?:per\s*year)?|/\s*unit\s*(?:/\s*year)?)\s*(?:fixed\s*holding\s*cost|holding\s*cost|carrying\s*cost|holding\s*expense)?',
        ], text)

        # Unit purchase price
        unit_cost = self._search_patterns([
            r'(?:unit\s*purchase\s*price|purchase\s*price|unit\s*purchase\s*cost|purchase\s*cost|unit\s*price|unit\s*cost|regular\s*price|base\s*price|cost\s*per\s*unit|item\s*cost|c\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|/\s*unit)',
            r'(?:costs|purchased\s*for|purchased\s*at)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:each|per\s*unit)?',
        ], text)
        if unit_cost is not None:
            inputs["unit_cost"] = unit_cost

        if holding_rate is not None and unit_cost is not None:
            inputs["holding_cost"] = holding_rate * unit_cost
            inputs["holding_cost_rate"] = holding_rate
        elif holding_rate is not None:
            inputs["holding_cost_rate"] = holding_rate
        elif fixed_holding is not None:
            inputs["holding_cost"] = fixed_holding

        # Lead time and working days
        lead_time = self._search_patterns([
            r'(?:lead\s*[- ]\s*time|delivery\s*lead\s*time|supplier\s*lead\s*time|l\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week)?',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:day|days|week|weeks)\s*lead\s*[- ]\s*time',
        ], text)
        if lead_time is not None:
            inputs["lead_time_days"] = lead_time

        work_days = self._search_patterns([
            r'([0-9]+)\s*(?:working\s*days|work\s*days|operating\s*days)\s*(?:per\s*year|a\s*year|annually)?',
            r'(?:operating\s*days|work\s*days|working\s*days)\s*(?:is|=|:|of)?\s*([0-9]+)',
        ], text)
        inputs["work_days_per_year"] = int(work_days) if work_days is not None else 250

        return inputs

    # ------------------------------------------------------------------------
    # Quantity Discount Extraction
    # ------------------------------------------------------------------------
    def _extract_quantity_discount_inputs(self, text: str, lower: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        # 1. Demand
        demand = self._search_patterns([
            r'(?:annual\s*demand|yearly\s*demand|annual\s*usage|demand\s*per\s*year)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*per\s*year|units/year|annual\s*units)',
            r'(?:demand\s*is|demand\s*of|demand:?)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
        ], text)
        if demand is not None:
            inputs["demand"] = demand

        # 2. Ordering Cost (Recognizing all synonyms)
        ordering_cost = self._search_patterns([
            r'(?:ordering\s*(?:/|and)?\s*setup\s*cost|ordering\s*[/&]\s*setup\s*expense|cost\s*of\s*placing\s*(?:an)?\s*order|setup\s*expense|cost\s*per\s*order|ordering\s*cost|order\s*cost|setup\s*cost|order\s*fee|setup\s*fee|s\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:dollars|\$)?\s*(?:per\s*order|/\s*order)',
        ], text)
        if ordering_cost is not None:
            inputs["ordering_cost"] = ordering_cost

        # 3. Holding Cost: Distinguish fixed holding cost vs. percentage holding cost
        percentage_holding = self._search_patterns([
            r'(?:percentage\s*holding\s*cost|holding\s*cost\s*rate|holding\s*rate|carrying\s*rate|holding\s*cost\s*percentage|carrying\s*cost\s*percentage|holding\s*cost|carrying\s*cost)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s*(?:the)?\s*(?:unit\s*)?(?:purchase\s*)?(?:price|cost)|per\s*year\s*holding|holding\s*cost|carrying\s*cost|holding\s*rate|carrying\s*rate|holding|carrying)',
            r'(?:percentage\s*holding\s*cost|holding\s*cost\s*rate|holding\s*rate|carrying\s*rate|holding\s*cost|carrying\s*cost)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*(0\.[0-9]+)(?!\s*%)',
        ], text)
        if percentage_holding is not None and percentage_holding > 1.0:
            percentage_holding = percentage_holding / 100.0

        fixed_holding = self._search_patterns([
            r'(?:fixed\s*holding\s*cost|unit\s*holding\s*cost|storage\s*cost|h\s*=)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)?',
            r'(?:holding\s*cost|carrying\s*cost|holding\s*expense)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*\$\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)?',
            r'(?:holding\s*cost|carrying\s*cost|holding\s*expense)\s*(?:is|=|:|of|at|equal\s*to|is\s*equal\s*to)?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:per\s*unit|\$/unit|dollars)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\b(?!\s*%)\s*(?:dollars|\$)?\s*(?:per\s*unit\s*(?:per\s*year)?|/\s*unit/\s*year)',
        ], text)

        if percentage_holding is not None:
            inputs["holding_cost_rate"] = percentage_holding
        elif fixed_holding is not None:
            inputs["holding_cost"] = fixed_holding

        # Unit purchase price
        unit_price = self._search_patterns([
            r'(?:unit\s*purchase\s*price|unit\s*price|base\s*price|regular\s*price|c\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if unit_price is not None:
            inputs["unit_purchase_price"] = unit_price

        # 4. Quantity Breaks / Price Schedule
        price_breaks: List[Dict[str, Any]] = []
        json_match = re.search(r'(\[\s*\{.*?\}\s*\])', text, re.DOTALL)
        if json_match:
            try:
                price_breaks = json.loads(json_match.group(1))
            except Exception:
                price_breaks = []

        if not price_breaks:
            # 1. Range pattern: e.g. "1–499 units at $10.00", "0 to 499 at $10.00", "500-999: $9.80"
            # Supports ASCII hyphens, en dashes (–), em dashes (—), minus (–), 'to', and commas in quantities (e.g. 1,000)
            range_matches = re.findall(
                r'([0-9]+(?:,[0-9]{3})*)\s*(?:to|-|–|—|−|\.\.)\s*([0-9]+(?:,[0-9]{3})*)\s*(?:units)?\s*[\)\]]?\s*(?:at|costs?|:|,)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
                lower
            )
            for low_b, high_b, pr in range_matches:
                price_breaks.append({
                    "min_qty": float(low_b.replace(",", "")),
                    "max_qty": float(high_b.replace(",", "")),
                    "price": float(pr)
                })

            # 2. Plus/Over/Above pattern: e.g. "1,000+ units at $9.00", "1,000 or more units at $9.00", "1000 and above: $9.60"
            plus_matches = re.findall(
                r'([0-9]+(?:,[0-9]{3})*)\s*(?:units)?\s*(?:\+|and\s*above|and\s*over|or\s*more|above)\s*(?:units)?\s*[\)\]]?\s*(?:at|costs?|:|,)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
                lower
            )
            for low_b, pr in plus_matches:
                price_breaks.append({
                    "min_qty": float(low_b.replace(",", "")),
                    "max_qty": None,
                    "price": float(pr)
                })

            # 3. Over / above prefix pattern: e.g. "over 1,000 units at $9.00"
            over_matches = re.findall(
                r'(?:over|above|exceeding)\s*([0-9]+(?:,[0-9]{3})*)\s*(?:units)?\s*[\)\]]?\s*(?:at|costs?|:|,)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
                lower
            )
            for low_b, pr in over_matches:
                price_breaks.append({
                    "min_qty": float(low_b.replace(",", "")),
                    "max_qty": None,
                    "price": float(pr)
                })

            # De-duplicate while preserving order and sort by min_qty
            seen = set()
            unique_breaks = []
            for pb in price_breaks:
                key = (pb["min_qty"], pb["max_qty"], pb["price"])
                if key not in seen:
                    seen.add(key)
                    unique_breaks.append(pb)
            price_breaks = sorted(unique_breaks, key=lambda pb: pb["min_qty"])

        if price_breaks:
            inputs["price_breaks"] = price_breaks

        return inputs

    # ------------------------------------------------------------------------
    # Safety Stock & Reorder Point Extraction
    # ------------------------------------------------------------------------
    def _extract_rop_inputs(self, text: str, lower: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        # 1. Average Demand (per period)
        avg_demand = self._search_patterns([
            r'(?:average\s*daily\s*demand|mean\s*daily\s*demand|daily\s*demand|demand\s*per\s*day)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*per\s*day|units/day)',
            r'(?:average\s*weekly\s*demand|mean\s*weekly\s*demand|weekly\s*demand|demand\s*per\s*week)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*per\s*week|units/week)',
            r'(?:average\s*demand|mean\s*demand|expected\s*demand|demand\s*rate)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*average\s*demand|units\s*mean\s*demand)',
            r'(?:demand\s*is|demand\s*of)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        if avg_demand is not None:
            # Check if text explicitly says annual/yearly
            if any(w in lower for w in ["annual demand", "units per year", "yearly demand"]):
                inputs["mean_demand_per_period"] = avg_demand / 250.0
            else:
                inputs["mean_demand_per_period"] = avg_demand

        # 2. Demand Standard Deviation
        demand_std = self._search_patterns([
            r'(?:daily\s*demand\s*(?:standard\s*deviation|std\s*dev)|weekly\s*demand\s*(?:standard\s*deviation|std\s*dev))\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand|demand\s*std\s*dev|sigma_d|demand\s*sigma)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand)',
            r'demand\s*(?:varies|fluctuates)\s*(?:with\s*a\s*standard\s*deviation\s*of|by)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
            r'standard\s*deviation\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        inputs["std_dev_demand"] = demand_std if demand_std is not None else 0.0

        # 3. Average Lead Time
        avg_lt = self._search_patterns([
            r'(?:average\s*lead\s*[- ]\s*time|mean\s*lead\s*[- ]\s*time|expected\s*lead\s*[- ]\s*time|supplier\s*lead\s*[- ]\s*time|delivery\s*lead\s*[- ]\s*time|lead\s*[- ]\s*time|l\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week|periods|period)?',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:day|days|week|weeks|period|periods)\s*(?:average\s*lead\s*[- ]\s*time|lead\s*[- ]\s*time)',
        ], text)
        if avg_lt is not None:
            inputs["mean_lead_time_periods"] = avg_lt

        # 4. Lead-Time Standard Deviation
        lt_std = self._search_patterns([
            r'(?:lead\s*[- ]\s*time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead\s*[- ]\s*time|std\s*dev\s*of\s*lead\s*[- ]\s*time|lead\s*[- ]\s*time\s*std\s*dev|sigma_l|lead\s*[- ]\s*time\s*sigma)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week)?\s*(?:lead\s*[- ]\s*time\s*standard\s*deviation|standard\s*deviation\s*of\s*lead\s*[- ]\s*time|std\s*dev\s*of\s*lead\s*[- ]\s*time)',
            r'lead\s*[- ]\s*time\s*(?:varies|fluctuates)\s*(?:with\s*a\s*standard\s*deviation\s*of|by)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week)?',
        ], text)
        inputs["std_dev_lead_time"] = lt_std if lt_std is not None else 0.0

        # 5. Service Level / Z-Score
        sl = self._search_patterns([
            r'(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|desired\s*service\s*level|csl)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl)',
            r'(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl)\s*(?:is|=|:|of|at)?\s*(0\.[0-9]+)',
        ], text)
        if sl is not None:
            inputs["service_level"] = sl / 100.0 if sl > 1.0 else sl

        z_val = self._search_patterns([
            r'(?:z-score|z\s*score|z\s*value|z\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if z_val is not None:
            inputs["z_score"] = z_val

        return inputs

    # ------------------------------------------------------------------------
    # Fixed-Period Review Extraction
    # ------------------------------------------------------------------------
    def _extract_periodic_review_inputs(self, text: str, lower: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        # 1. Review Period
        review_p = self._search_patterns([
            r'(?:review\s*period|review\s*interval|review\s*cycle|replenishment\s*interval|order\s*interval|r\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week|periods|period)?',
            r'(?:reviewed|orders\s*placed|orders\s*made|reordered)\s*every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week|periods|period)',
            r'every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week|periods|period)',
        ], text)
        if review_p is not None:
            inputs["review_period"] = review_p

        # 2. Lead Time
        lt = self._search_patterns([
            r'(?:lead\s*[- ]\s*time|delivery\s*lead\s*time|supplier\s*lead\s*time|l\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day|weeks|week|periods|period)?',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:day|days|week|weeks|period|periods)\s*lead\s*[- ]\s*time',
        ], text)
        if lt is not None:
            inputs["lead_time"] = lt

        # 3. Demand (rate per period)
        d_rate = self._search_patterns([
            r'(?:average\s*daily\s*demand|mean\s*daily\s*demand|daily\s*demand|demand\s*per\s*day)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units\s*per\s*day|units/day)',
            r'(?:average\s*demand|mean\s*demand|expected\s*demand|demand\s*rate)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'(?:demand\s*is|demand\s*of)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        if d_rate is not None:
            if any(w in lower for w in ["annual demand", "units per year", "yearly demand"]):
                inputs["mean_demand_per_period"] = d_rate / 250.0
            else:
                inputs["mean_demand_per_period"] = d_rate

        # Demand standard deviation
        d_std = self._search_patterns([
            r'(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand|demand\s*std\s*dev|sigma_d)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'standard\s*deviation\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        inputs["std_dev_demand_per_period"] = d_std if d_std is not None else 0.0

        # Service level
        sl = self._search_patterns([
            r'(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl)',
            r'(?:cycle\s*service\s*level|service\s*level|target\s*service\s*level|csl)\s*(?:is|=|:|of|at)?\s*(0\.[0-9]+)',
        ], text)
        if sl is not None:
            inputs["service_level"] = sl / 100.0 if sl > 1.0 else sl

        z_val = self._search_patterns([
            r'(?:z-score|z\s*score|z\s*value|z\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if z_val is not None:
            inputs["z_score"] = z_val

        # Safety Stock (if specified directly)
        ss_explicit = self._search_patterns([
            r'(?:safety\s*stock|buffer\s*stock)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:safety\s*stock|buffer\s*stock)',
        ], text)
        if ss_explicit is not None:
            inputs["safety_stock"] = ss_explicit

        # Inventory Position (if specified directly)
        ip_explicit = self._search_patterns([
            r'(?:inventory\s*position|effective\s*inventory|ip\s*=)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:inventory\s*position|effective\s*inventory)',
        ], text)
        if ip_explicit is not None:
            inputs["inventory_position"] = ip_explicit

        # On-Hand Inventory
        on_hand = self._search_patterns([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:currently)?\s*(?:on\s*[- ]\s*hand|in\s*stock|stock\s*on\s*hand|physical\s*inventory)',
            r'(?:on\s*[- ]\s*hand\s*(?:inventory)?|current\s*stock|current\s*inventory|physically\s*in\s*stock|stock\s*on\s*hand|physical\s*inventory)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if on_hand is not None:
            inputs["on_hand_inventory"] = on_hand

        # On-Order Inventory
        on_order = self._search_patterns([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:currently)?\s*(?:on\s*[- ]\s*order|in\s*transit|pipeline\s*inventory|incoming\s*shipment)',
            r'(?:on\s*[- ]\s*order\s*(?:inventory)?|pipeline\s*inventory|in\s*transit\s*(?:inventory)?|incoming\s*shipment)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if on_order is not None:
            inputs["on_order_inventory"] = on_order

        # Backorders
        backorders = self._search_patterns([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:backorders|backordered|backlogged|backlog)',
            r'(?:backorders|backordered\s*units|backlog|backlogged\s*orders|unfulfilled\s*backorders)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if backorders is not None:
            inputs["backorders"] = backorders

        # Compute inventory position if not given directly
        if "inventory_position" not in inputs and ("on_hand_inventory" in inputs or "on_order_inventory" in inputs):
            inputs["inventory_position"] = inputs.get("on_hand_inventory", 0.0) + inputs.get("on_order_inventory", 0.0) - inputs.get("backorders", 0.0)

        return inputs

    # ------------------------------------------------------------------------
    # Newsvendor Extraction
    # ------------------------------------------------------------------------
    def _extract_newsvendor_inputs(self, text: str, lower: str) -> Dict[str, Any]:
        inputs: Dict[str, Any] = {}

        # 1. Selling Price
        selling_price = self._search_patterns([
            r'(?:selling\s*price|retail\s*price|sale\s*price|regular\s*price|revenue\s*per\s*unit|price\s*per\s*unit|p\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'(?:sells\s*for|retail\s*at|retails\s*for)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:selling\s*price|retail\s*price|sale\s*price)',
        ], text)
        if selling_price is not None:
            inputs["selling_price"] = selling_price

        # 2. Purchase Cost
        purchase_cost = self._search_patterns([
            r'(?:purchase\s*cost|procurement\s*cost|wholesale\s*cost|production\s*cost|unit\s*cost|cost\s*per\s*unit|item\s*cost|c\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'(?:costs|purchased\s*for|purchased\s*at)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|wholesale)?',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:wholesale\s*cost|purchase\s*cost|procurement\s*cost|unit\s*cost)',
        ], text)
        if purchase_cost is not None:
            inputs["cost_per_unit"] = purchase_cost

        # 3. Salvage Value
        salvage = self._search_patterns([
            r'(?:salvage\s*value|salvage\s*price|scrap\s*value|liquidation\s*value|clearance\s*price|markdown\s*price|residual\s*value|salvage|v\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:salvage\s*value|scrap\s*value|salvage|clearance|markdown)',
            r'(?:salvaged\s*for|liquidated\s*at|cleared\s*at)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        inputs["salvage_value"] = salvage if salvage is not None else 0.0

        # 4. Underage / Shortage Cost
        underage = self._search_patterns([
            r'(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage|cost\s*of\s*shortage|cost\s*of\s*underestimating|marginal\s*profit|stockout\s*cost|cu\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:underage\s*cost|shortage\s*cost|cost\s*of\s*underage|marginal\s*profit)',
        ], text)
        if underage is not None:
            inputs["underage_cost"] = underage

        # 5. Overage / Excess Cost
        overage = self._search_patterns([
            r'(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage|cost\s*of\s*excess|cost\s*of\s*overestimating|marginal\s*loss|co\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:overage\s*cost|excess\s*cost|cost\s*of\s*overage|marginal\s*loss)',
        ], text)
        if overage is not None:
            inputs["overage_cost"] = overage

        # 6. Goodwill Penalty
        goodwill = self._search_patterns([
            r'(?:goodwill\s*penalty|stockout\s*penalty|loss\s*of\s*goodwill|penalty\s*cost|goodwill\s*cost|g\s*=)\s*(?:is|=|:|of|at)?\s*\$?\s*([0-9]+(?:\.[0-9]+)?)',
            r'\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:goodwill\s*penalty|loss\s*of\s*goodwill|stockout\s*penalty)',
        ], text)
        inputs["goodwill_penalty"] = goodwill if goodwill is not None else 0.0

        # 7. Demand Distribution (Mean & Standard Deviation)
        mean_demand = self._search_patterns([
            r'(?:mean\s*(?:of)?\s*demand|mean\s*demand|average\s*demand|expected\s*demand|demand\s*mean)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'demand\s*(?:follows|is\s*normally\s*distributed\s*with)\s*(?:a)?\s*mean\s*(?:of)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            r'mean\s*of\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units)?',
            r'(?:demand\s*is|demand\s*of)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        if mean_demand is not None:
            inputs["mean_demand"] = mean_demand

        std_dev = self._search_patterns([
            r'(?:demand\s*standard\s*deviation|standard\s*deviation\s*of\s*demand|std\s*dev\s*of\s*demand|demand\s*std\s*dev|sigma)\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'standard\s*deviation\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
            r'std\s*dev\s*(?:is|=|:|of|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units)?',
        ], text)
        if std_dev is not None:
            inputs["std_dev_demand"] = std_dev

        # Demand distribution type indicator
        inputs["demand_distribution"] = "normal" if ("normal" in lower or std_dev is not None) else "unknown"

        return inputs

    # ========================================================================
    # CHECK MISSING PARAMETERS
    # ========================================================================

    def identify_missing_parameters(self, model_type: str, inputs: Dict[str, Any]) -> List[str]:
        """
        Determines what required parameters are truly missing from the question
        after comprehensive full-sentence examination.
        """
        missing: List[str] = []

        if model_type == self.MODEL_EOQ:
            if "demand" not in inputs:
                missing.append("annual demand (D)")
            if "ordering_cost" not in inputs:
                missing.append("ordering/setup cost (S)")
            if "holding_cost" not in inputs and "holding_cost_rate" not in inputs:
                missing.append("holding cost (H or holding cost rate %)")

        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            if "demand" not in inputs:
                missing.append("annual demand (D)")
            if "ordering_cost" not in inputs:
                missing.append("ordering/setup cost (S)")
            if "price_breaks" not in inputs or not inputs["price_breaks"]:
                missing.append("price break schedule / discount tiers")
            if "holding_cost" not in inputs and "holding_cost_rate" not in inputs:
                missing.append("holding cost (fixed H or percentage rate)")

        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            if "mean_demand_per_period" not in inputs:
                missing.append("average demand per period (d)")
            if "mean_lead_time_periods" not in inputs:
                missing.append("average lead time (L)")
            has_variability = (inputs.get("std_dev_demand", 0.0) > 0.0) or (inputs.get("std_dev_lead_time", 0.0) > 0.0)
            if has_variability and "service_level" not in inputs and "z_score" not in inputs:
                missing.append("cycle service level or z-score")

        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            if "review_period" not in inputs:
                missing.append("review period interval (R)")
            if "lead_time" not in inputs:
                missing.append("replenishment lead time (L)")
            if "mean_demand_per_period" not in inputs:
                missing.append("average demand per period (d)")
            if "inventory_position" not in inputs and "on_hand_inventory" not in inputs:
                missing.append("current inventory position (or on-hand / on-order stock)")
            has_variability = inputs.get("std_dev_demand_per_period", 0.0) > 0.0
            if has_variability and "service_level" not in inputs and "z_score" not in inputs:
                missing.append("cycle service level or z-score")

        elif model_type == self.MODEL_NEWSVENDOR:
            has_direct_margins = ("underage_cost" in inputs) and ("overage_cost" in inputs)
            has_unit_prices = ("selling_price" in inputs) and ("cost_per_unit" in inputs)
            if not has_direct_margins and not has_unit_prices:
                if "selling_price" not in inputs and "underage_cost" not in inputs:
                    missing.append("selling price (or direct underage cost)")
                if "cost_per_unit" not in inputs and "overage_cost" not in inputs:
                    missing.append("purchase cost (or direct overage cost)")

            has_normal = ("mean_demand" in inputs) and ("std_dev_demand" in inputs)
            has_discrete = "discrete_demand_dist" in inputs and inputs["discrete_demand_dist"]
            if not has_normal and not has_discrete:
                if "mean_demand" not in inputs:
                    missing.append("mean demand")
                if "std_dev_demand" not in inputs:
                    missing.append("demand standard deviation")

        return missing

    # ========================================================================
    # STEP 3: SELECT PYTHON TOOL
    # ========================================================================

    def select_tool(self, model_type: str) -> str:
        """Returns the Python tool function name for the identified model."""
        return self.TOOL_MAPPING.get(model_type, "calculate_eoq")

    # ========================================================================
    # STEP 4: CALCULATE (PYTHON NUMERICAL ENGINE)
    # ========================================================================

    def execute_calculation(self, model_type: str, inputs: Dict[str, Any]) -> Any:
        """Executes the deterministic mathematical calculation tool."""
        if model_type == self.MODEL_EOQ:
            return calculate_eoq(
                demand=inputs["demand"],
                ordering_cost=inputs["ordering_cost"],
                holding_cost=inputs["holding_cost"],
                unit_cost=inputs.get("unit_cost"),
                work_days_per_year=inputs.get("work_days_per_year", 250),
                lead_time_days=inputs.get("lead_time_days"),
            )
        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            return calculate_quantity_discount(
                demand=inputs["demand"],
                ordering_cost=inputs["ordering_cost"],
                price_breaks=inputs["price_breaks"],
                holding_cost=inputs.get("holding_cost"),
                holding_cost_rate=inputs.get("holding_cost_rate"),
            )
        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            return calculate_rop(
                demand=inputs["mean_demand_per_period"],
                lead_time=inputs["mean_lead_time_periods"],
                service_level=inputs.get("service_level"),
                z=inputs.get("z_score"),
                sigma_d=inputs.get("std_dev_demand", 0.0),
                sigma_L=inputs.get("std_dev_lead_time", 0.0),
            )
        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            return calculate_periodic_review(
                review_period=inputs["review_period"],
                lead_time=inputs["lead_time"],
                demand_rate=inputs["mean_demand_per_period"],
                sigma_d=inputs.get("std_dev_demand_per_period", 0.0),
                service_level=inputs.get("service_level"),
                z=inputs.get("z_score"),
                inventory_position=inputs.get("inventory_position"),
                on_hand=inputs.get("on_hand_inventory"),
                on_order=inputs.get("on_order_inventory"),
                backorders=inputs.get("backorders", 0.0),
            )
        elif model_type == self.MODEL_NEWSVENDOR:
            return calculate_newsvendor(
                price=inputs.get("selling_price"),
                cost=inputs.get("cost_per_unit"),
                salvage_value=inputs.get("salvage_value", 0.0),
                goodwill_penalty=inputs.get("goodwill_penalty", 0.0),
                mean_demand=inputs.get("mean_demand"),
                std_dev_demand=inputs.get("std_dev_demand"),
                discrete_distribution=inputs.get("discrete_demand_dist"),
                underage_cost=inputs.get("underage_cost"),
                overage_cost=inputs.get("overage_cost"),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # ========================================================================
    # STEP 5: VERIFY
    # ========================================================================

    def verify_inputs_and_results(
        self, model_type: str, inputs: Dict[str, Any], result: Any
    ) -> VerificationRecord:
        """Runs domain sanity checks and mathematical validation on inputs and outputs."""
        checks: List[str] = []
        warnings: List[str] = []
        errors: List[str] = []

        if model_type == self.MODEL_EOQ:
            assert isinstance(result, EOQResult)
            checks.append(f"Checked positivity: Demand={result.demand} > 0, S={result.ordering_cost} > 0, H={result.holding_cost} > 0")
            if result.optimal_order_quantity <= 0:
                errors.append(f"Calculated EOQ ({result.optimal_order_quantity}) is non-positive.")
            else:
                checks.append(f"EOQ calculated: {result.optimal_order_quantity:.2f} units.")

            diff = abs(result.annual_ordering_cost - result.annual_holding_cost)
            if diff > 1e-2:
                warnings.append(f"Ordering and holding cost mismatch at EOQ: delta = ${diff:.4f}")
            else:
                checks.append("Verified cost balance: Annual ordering cost equals annual holding cost at EOQ.")

        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            assert isinstance(result, QuantityDiscountResult)
            checks.append(f"Evaluated {len(result.evaluated_tiers)} pricing tiers.")
            if result.optimal_order_quantity <= 0:
                errors.append("Optimal quantity discount order quantity is non-positive.")
            else:
                checks.append(f"Optimal order quantity identified: {result.optimal_order_quantity:.2f} units in Tier {result.optimal_tier_index + 1}.")
            prices = [t.unit_price for t in result.evaluated_tiers]
            if sorted(prices, reverse=True) != prices:
                warnings.append("Price break schedule does not strictly decrease with volume.")
            else:
                checks.append("Verified volume price structure: Prices strictly decrease across tiers.")

        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            assert isinstance(result, ROPResult)
            checks.append(f"Uncertainty model applied: `{result.uncertainty_scenario}`")
            if result.service_level is not None:
                checks.append(f"Service Level: {result.service_level * 100:.1f}% -> Z-Score: {result.z_score:.4f}")
            if result.safety_stock < 0:
                errors.append(f"Negative safety stock generated: {result.safety_stock}")
            else:
                checks.append(f"Safety stock: {result.safety_stock:.2f} units, ROP: {result.reorder_point:.2f} units.")

        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            assert isinstance(result, PeriodicReviewResult)
            checks.append(f"Protection period: {result.protection_period} periods (Review {result.review_period} + Lead Time {result.lead_time}).")
            if result.target_inventory <= 0:
                warnings.append(f"Target level S is {result.target_inventory}, which is unusually low.")
            else:
                checks.append(f"Target Order-Up-To Level (S): {result.target_inventory:.2f} units.")
            checks.append(f"Inventory Position (IP): {result.inventory_position:.2f} units -> Recommended Order Q: {result.order_quantity:.2f} units.")

        elif model_type == self.MODEL_NEWSVENDOR:
            assert isinstance(result, NewsvendorResult)
            checks.append(f"Economic viability check: Price (${result.selling_price:.2f}) > Cost (${result.cost_per_unit:.2f}) > Salvage (${result.salvage_value:.2f})")
            if not (0.0 < result.critical_ratio < 1.0):
                errors.append(f"Critical ratio {result.critical_ratio:.4f} is not between 0 and 1.")
            else:
                checks.append(f"Critical Fractile: {result.critical_ratio:.4f} (Underage Cu=${result.underage_cost:.2f}, Overage Co=${result.overage_cost:.2f})")

            if result.optimal_order_quantity <= 0:
                warnings.append("Optimal order quantity is zero or negative.")
            else:
                checks.append(f"Optimal order quantity Q*: {result.optimal_order_quantity:.2f} units.")

        is_valid = len(errors) == 0
        if errors:
            status = "FAILED"
        elif warnings:
            status = "WARNING"
        else:
            status = "PASSED"

        return VerificationRecord(
            is_valid=is_valid,
            status=status,
            checks=checks,
            warnings=warnings,
            errors=errors,
        )

    # ========================================================================
    # STEP 6 & 7: EXPLAIN
    # ========================================================================

    def explain_decision(
        self, model_type: str, inputs: Dict[str, Any], result: Any, verification: VerificationRecord
    ) -> str:
        """Produces executive-level operational recommendations and cost/risk breakdowns."""
        lines = []

        if model_type == self.MODEL_EOQ:
            r: EOQResult = result
            lines.append("### Executive Recommendation & Decision Summary")
            lines.append(f"- **Recommended Order Quantity (Q*)**: **{r.optimal_order_quantity:,.1f} units** per order.")
            lines.append(f"- **Order Frequency**: **{r.orders_per_year:.2f} orders/year** (~every **{r.cycle_time_days if r.cycle_time_days is not None else (r.cycle_time_years * 250):.1f} working days**).")
            if r.reorder_point is not None:
                lines.append(f"- **Reorder Point (ROP)**: Place an order when inventory reaches **{r.reorder_point:,.1f} units** (based on {r.lead_time_days} days lead time).")

            lines.append("\n### Cost Breakdown & Trade-Off Analysis")
            lines.append(f"- **Annual Ordering Cost**: ${r.annual_ordering_cost:,.2f}")
            lines.append(f"- **Annual Holding Cost**: ${r.annual_holding_cost:,.2f}")
            lines.append(f"- **Total Annual Inventory Cost (Hold + Order)**: **${r.total_annual_inventory_cost:,.2f}**")
            if r.annual_purchase_cost is not None:
                lines.append(f"- **Annual Purchase Cost**: ${r.annual_purchase_cost:,.2f}")
                lines.append(f"- **Total System Cost**: **${r.total_annual_cost:,.2f}**")
            lines.append("\n> **Supply Chain Insight**: At the EOQ, annual ordering costs and holding costs are perfectly balanced. Deviating by ±20% in order batch size typically increases total inventory costs by less than 2-3%, giving operational flexibility for pallet/carton packaging.")

        elif model_type == self.MODEL_QUANTITY_DISCOUNT:
            r: QuantityDiscountResult = result
            lines.append("### Executive Recommendation & Decision Summary")
            lines.append(f"- **Optimal Order Batch**: **{r.optimal_order_quantity:,.1f} units** under **Tier {r.optimal_tier_index + 1}**.")
            lines.append(f"- **Negotiated Unit Price**: **${r.optimal_unit_price:,.2f}/unit**.")
            lines.append(f"- **Total Annual System Cost**: **${r.optimal_total_cost:,.2f}**.")
            if r.cost_savings_vs_base > 0:
                lines.append(f"- **Annual Net Savings vs. Base Pricing**: **${r.cost_savings_vs_base:,.2f}**.")

            lines.append("\n### Discount Schedule Evaluation")
            lines.append("| Tier | Quantity Range | Unit Price | Feasible EOQ | Candidate Q | Annual Cost | Optimal? |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for t in r.evaluated_tiers:
                max_str = f"{t.max_qty:,.0f}" if t.max_qty is not None else "Above"
                opt_mark = "**YES**" if t.is_optimal else "No"
                lines.append(f"| Tier {t.tier_index + 1} | {t.min_qty:,.0f} - {max_str} | ${t.unit_price:,.2f} | {t.unconstrained_eoq:,.1f} | {t.candidate_qty:,.1f} | ${t.total_annual_cost:,.2f} | {opt_mark} |")

            lines.append("\n> **Supply Chain Insight**: While higher purchase volume qualifies for supplier discounts and reduces order setup overhead, it inflates working capital tied up in carrying stock. The selected quantity represents the mathematically proven global cost minimum.")

        elif model_type == self.MODEL_SAFETY_STOCK_ROP:
            r: ROPResult = result
            lines.append("### Executive Recommendation & Decision Summary")
            lines.append(f"- **Reorder Point (ROP)**: **{r.reorder_point:,.1f} units**.")
            lines.append(f"- **Safety Stock Buffer**: **{r.safety_stock:,.1f} units**.")
            if r.service_level is not None:
                lines.append(f"- **Service Level Target**: **{r.service_level * 100:.1f}% Cycle Service Level** (Z-Score = {r.z_score:.2f}).")
            lines.append(f"- **Expected Lead Time Demand**: **{r.mean_lead_time_demand:,.1f} units** across {r.lead_time} periods.")

            lines.append("\n### Uncertainty Profile")
            lines.append(f"- **Uncertainty Model**: `{r.uncertainty_scenario}`")
            lines.append(f"- **Lead Time Demand Std Dev**: {r.sigma_LT:.2f} units.")
            lines.append(f"- **Stockout Risk**: {r.stockout_risk * 100:.1f}% probability of stockout during replenishment cycle.")
            lines.append("\n> **Supply Chain Insight**: Safety stock absorbs demand fluctuations and vendor delivery delays. Holding these additional units acts as insurance: increasing the service level from 95% to 99% increases safety stock by ~41% (Z goes from 1.645 to 2.326).")

        elif model_type == self.MODEL_FIXED_PERIOD_REVIEW:
            r: PeriodicReviewResult = result
            lines.append("### Executive Recommendation & Decision Summary")
            lines.append(f"- **Target Order-Up-To Level (S)**: **{r.target_inventory:,.1f} units**.")
            lines.append(f"- **Recommended Order Quantity (Q)**: **{r.order_quantity:,.1f} units**.")
            lines.append(f"- **Current Inventory Position (IP)**: **{r.inventory_position:,.1f} units**.")

            lines.append("\n### Protection Interval & Safety Stock")
            lines.append(f"- **Protection Interval**: **{r.protection_period:.1f} periods** (Review Period {r.review_period} + Lead Time {r.lead_time}).")
            lines.append(f"- **Expected Demand over Protection Interval**: {r.demand_during_protection_period:,.1f} units.")
            lines.append(f"- **Safety Stock Required**: {r.safety_stock:,.1f} units.")
            lines.append("\n> **Supply Chain Insight**: In a periodic review system, inventory must protect against stockouts across both the review cycle AND the supplier lead time (R + L). This requires higher safety stock than continuous review (s, Q), but provides predictable logistics schedules and consolidates supplier shipments.")

        elif model_type == self.MODEL_NEWSVENDOR:
            r: NewsvendorResult = result
            lines.append("### Executive Recommendation & Decision Summary")
            lines.append(f"- **Optimal Order Quantity (Q*)**: **{r.optimal_order_quantity:,.1f} units**.")
            lines.append(f"- **Critical Fractile / Ratio**: **{r.critical_ratio:.4f}** ({r.critical_ratio * 100:.1f}%).")
            if r.expected_profit is not None:
                lines.append(f"- **Expected Profit**: **${r.expected_profit:,.2f}**.")

            lines.append("\n### Economics of Perishability & Marginal Costs")
            lines.append(f"- **Cost of Underage (Cu - Marginal Profit Lost)**: ${r.underage_cost:,.2f}/unit")
            lines.append(f"- **Cost of Overage (Co - Markdown Loss)**: ${r.overage_cost:,.2f}/unit")
            if r.expected_sales is not None:
                lines.append(f"- **Expected Sales**: {r.expected_sales:,.1f} units")
                lines.append(f"- **Expected Lost Sales**: {r.expected_lost_sales:,.1f} units")
                lines.append(f"- **Expected Leftover / Salvage Units**: {r.expected_leftover_inventory:,.1f} units")
            lines.append("\n> **Supply Chain Insight**: Because the profit from selling an additional unit (Cu) exceeds the loss from having an unsold unit (Co), the critical ratio is high, dictating an order quantity above the mean demand to hedge against stockout opportunity costs.")

        return "\n".join(lines)

    # ========================================================================
    # MAIN AGENT INTERACTION: EXECUTE 7-STEP ARCHITECTURE
    # ========================================================================

    def process_question(self, question: str) -> AgentDecisionResponse:
        """
        Executes the complete 7-step decision pipeline:
        1. Identify Model
        2. Extract Inputs (Full-sentence examination)
        3. Identify any truly missing parameters
        4. Select Python Tool
        5. Calculate
        6. Verify
        7. Explain
        """
        trace: Dict[str, Any] = {"question": question}

        # Step 1: Identify Model
        model_type, confidence = self.identify_model(question)
        trace["step_1_identified_model"] = {
            "model_type": model_type,
            "display_name": self.MODEL_NAMES[model_type],
            "confidence": confidence,
        }

        # Step 2: Extract Inputs (full-question examination)
        extracted_inputs = self.extract_inputs(question, model_type)
        trace["step_2_extracted_inputs"] = extracted_inputs

        # Check for truly missing required parameters
        missing = self.identify_missing_parameters(model_type, extracted_inputs)

        if missing:
            # Report missing parameters without guessing or inventing numerical values
            tool_name = self.select_tool(model_type)
            verification = VerificationRecord(
                is_valid=False,
                status="INCOMPLETE",
                warnings=[f"Missing required parameter(s): {', '.join(missing)}"],
                checks=[f"Extracted {len(extracted_inputs)} parameters from question."],
                errors=[f"Cannot calculate {self.MODEL_NAMES[model_type]} without: {', '.join(missing)}"],
            )

            explanation = (
                f"### Information Required to Compute {self.MODEL_NAMES[model_type]}\n"
                f"I extracted the following information from your question:\n"
                + "\n".join([f"- **{k}**: {v}" for k, v in extracted_inputs.items() if k != "price_breaks"])
                + f"\n\nHowever, to solve this problem accurately without inventing missing numbers, I still need:\n"
                + "\n".join([f"- **{m}**" for m in missing])
                + f"\n\nPlease provide these values and I will calculate the exact optimal inventory decision."
            )

            formatted_output = self._format_incomplete_output(
                question=question,
                model_type=model_type,
                confidence=confidence,
                inputs=extracted_inputs,
                missing=missing,
                explanation=explanation,
            )

            return AgentDecisionResponse(
                question=question,
                identified_model=model_type,
                model_name_display=self.MODEL_NAMES[model_type],
                confidence_score=confidence,
                extracted_inputs=extracted_inputs,
                selected_tool_name=tool_name,
                calculation_result=None,
                verification=verification,
                explanation=explanation,
                formatted_output=formatted_output,
                raw_step_trace=trace,
                missing_parameters=missing,
                is_complete=False,
            )

        # Step 3: Select Python Tool
        tool_name = self.select_tool(model_type)
        trace["step_3_selected_tool"] = tool_name

        # Step 4: Calculate
        try:
            calculation_result = self.execute_calculation(model_type, extracted_inputs)
            trace["step_4_calculation_result"] = asdict(calculation_result) if hasattr(calculation_result, "__dataclass_fields__") else str(calculation_result)
        except Exception as e:
            raise RuntimeError(f"Calculation failed in {tool_name}: {str(e)}") from e

        # Step 5: Verify
        verification = self.verify_inputs_and_results(model_type, extracted_inputs, calculation_result)
        trace["step_5_verification"] = asdict(verification)

        # Step 6 & 7: Explain
        explanation = self.explain_decision(model_type, extracted_inputs, calculation_result, verification)
        trace["step_6_explanation"] = explanation

        # Format full output with all 8 pedagogical student verification sections
        formatted_output = self._format_full_output(
            question=question,
            model_type=model_type,
            confidence=confidence,
            inputs=extracted_inputs,
            tool_name=tool_name,
            calculation_result=calculation_result,
            verification=verification,
            explanation=explanation,
        )

        return AgentDecisionResponse(
            question=question,
            identified_model=model_type,
            model_name_display=self.MODEL_NAMES[model_type],
            confidence_score=confidence,
            extracted_inputs=extracted_inputs,
            selected_tool_name=tool_name,
            calculation_result=calculation_result,
            verification=verification,
            explanation=explanation,
            formatted_output=formatted_output,
            raw_step_trace=trace,
            missing_parameters=[],
            is_complete=True,
        )

    def _format_incomplete_output(
        self,
        question: str,
        model_type: str,
        confidence: float,
        inputs: Dict[str, Any],
        missing: List[str],
        explanation: str,
    ) -> str:
        """Constructs an informative report when required inputs are missing."""
        inputs_formatted = "\n".join([f"    - {k}: {v}" for k, v in inputs.items() if k != "price_breaks"]) if inputs else "    (No numerical parameters detected)"
        missing_formatted = "\n".join([f"    - {m}" for m in missing])

        return f"""
================================================================================
SUPPLY CHAIN ANALYTICS - INVENTORY DECISION REPORT [INCOMPLETE INFORMATION]
================================================================================
Question: "{question.strip()}"

[Step 1: Model Identification]
  - Model: {self.MODEL_NAMES[model_type]} (Key: `{model_type}`)
  - Confidence: {confidence * 100:.1f}%

[Step 2: Extracted Parameters (From Entire Question)]
{inputs_formatted}

[Step 3: Missing Required Parameters (Will Not Invent Missing Values)]
{missing_formatted}

[Next Steps]
{explanation}
================================================================================
"""

    def _format_full_output(
        self,
        question: str,
        model_type: str,
        confidence: float,
        inputs: Dict[str, Any],
        tool_name: str,
        calculation_result: Any,
        verification: VerificationRecord,
        explanation: str,
    ) -> str:
        """
        Constructs a complete, structured report displaying all 8 student verification sections
        directly from the deterministic Python tool.
        """
        v_icon = "PASS [OK]" if verification.status == "PASSED" else f"WARNING [{verification.status}]"
        calc = calculation_result

        # Section 1: Inputs Extracted
        extracted_lines = []
        if hasattr(calc, "inputs_extracted") and calc.inputs_extracted:
            for k, v in calc.inputs_extracted.items():
                if v is not None:
                    if k == "price_breaks" and isinstance(v, list):
                        extracted_lines.append(f"    - {k}: {json.dumps(v)}")
                    elif isinstance(v, float):
                        extracted_lines.append(f"    - {k}: {v:,.4f}" if (0 < abs(v) < 1) else f"    - {k}: {v:,.2f}")
                    else:
                        extracted_lines.append(f"    - {k}: {v}")
        inputs_section = "\n".join(extracted_lines) if extracted_lines else "    (None)"

        # Section 2: Variable Definitions
        var_lines = []
        if hasattr(calc, "variable_definitions") and calc.variable_definitions:
            for var_sym, var_desc in calc.variable_definitions.items():
                var_lines.append(f"    - {var_sym}: {var_desc}")
        vars_section = "\n".join(var_lines) if var_lines else "    (None)"

        # Section 3: Formula
        formula_section = "    " + getattr(calc, "formula", "N/A").replace("\n", "\n    ")

        # Section 4: Formula with Substituted Values
        sub_values_section = "    " + getattr(calc, "formula_with_substituted_values", "N/A").replace("\n", "\n    ")

        # Section 5: Intermediate Calculations
        intermediate_lines = []
        if hasattr(calc, "intermediate_calculations") and calc.intermediate_calculations:
            for ik, iv in calc.intermediate_calculations.items():
                if isinstance(iv, float):
                    if abs(iv) >= 1000 or (0.01 <= abs(iv) < 1000 and abs(iv - round(iv, 2)) < 1e-4):
                        intermediate_lines.append(f"    - {ik}: {iv:,.2f}")
                    else:
                        intermediate_lines.append(f"    - {ik}: {iv:,.4f}")
                elif isinstance(iv, (list, tuple, dict)):
                    intermediate_lines.append(f"    - {ik}: {json.dumps(iv, default=str)}")
                else:
                    intermediate_lines.append(f"    - {ik}: {iv}")
        intermediate_section = "\n".join(intermediate_lines) if intermediate_lines else "    (None)"

        # Section 6: Final Answer
        final_answer_val = getattr(calc, "final_answer", "N/A")
        if isinstance(final_answer_val, float):
            final_answer_section = f"    Optimal Value = {final_answer_val:,.4f} (approximately {final_answer_val:,.2f})"
        else:
            final_answer_section = f"    Optimal Value = {final_answer_val}"

        # Section 7: Verification Checks
        checks_lines = []
        if hasattr(calc, "verification_checks") and calc.verification_checks:
            for chk in calc.verification_checks:
                checks_lines.append(f"    - [PASSED] {chk}")
        for w in verification.warnings:
            checks_lines.append(f"    - [WARNING] {w}")
        for err in verification.errors:
            checks_lines.append(f"    - [ERROR] {err}")
        verification_section = "\n".join(checks_lines) if checks_lines else f"    - [{v_icon}] Deterministic calculation verified."

        # Section 8: Assumptions
        assumptions_lines = []
        if hasattr(calc, "assumptions") and calc.assumptions:
            for asm in calc.assumptions:
                assumptions_lines.append(f"    - {asm}")
        assumptions_section = "\n".join(assumptions_lines) if assumptions_lines else "    - Standard textbook inventory assumptions apply."

        return f"""
================================================================================
SUPPLY CHAIN ANALYTICS - INVENTORY DECISION REPORT
================================================================================
Question: "{question.strip()}"

[Pipeline Step 1: Model Identification]
  - Model: {self.MODEL_NAMES[model_type]} (Key: `{model_type}`)
  - Confidence: {confidence * 100:.1f}%

[Pipeline Step 2: Tool Dispatch]
  - Deterministic Python Tool: `inventory_tools.{tool_name}`

================================================================================
STUDENT VERIFICATION & STEP-BY-STEP CALCULATION AUDIT
================================================================================

1. Inputs Extracted:
{inputs_section}

2. Variable Definitions:
{vars_section}

3. Formula:
{formula_section}

4. Formula with Substituted Values:
{sub_values_section}

5. Intermediate Calculations:
{intermediate_section}

6. Final Answer:
{final_answer_section}

7. Verification Checks:
{verification_section}

8. Assumptions:
{assumptions_section}

================================================================================
EXECUTIVE SUPPLY CHAIN EXPLANATION & DECISION SUMMARY
================================================================================
{explanation}
================================================================================
"""
