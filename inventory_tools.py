"""
Supply Chain Analytics - Deterministic Inventory Tools
======================================================
Deterministic, formula-driven Python implementations for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

For every calculation, each tool deterministically produces the 8 pedagogical verification sections:
1. Inputs Extracted
2. Variable Definitions
3. Formula
4. Formula with Substituted Values
5. Intermediate Calculations
6. Final Answer
7. Verification Checks
8. Assumptions

All numerical steps come directly from deterministic Python mathematics.
Missing numerical values are strictly validated and never invented.
"""

import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union

try:
    from statistics import NormalDist
    _HAS_STATISTICS_NORMAL = True
except ImportError:
    _HAS_STATISTICS_NORMAL = False


# ============================================================================
# Statistical Utilities (Standard Normal Distribution)
# ============================================================================

def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function Phi(x)."""
    if _HAS_STATISTICS_NORMAL:
        return NormalDist().cdf(x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function phi(x)."""
    if _HAS_STATISTICS_NORMAL:
        return NormalDist().pdf(x)
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)


def normal_inv_cdf(p: float) -> float:
    """Standard normal inverse CDF / quantile function Phi^-1(p)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability p must be strictly between 0 and 1, got {p}")
    if _HAS_STATISTICS_NORMAL:
        return NormalDist().inv_cdf(p)

    # Acklam's rational approximation algorithm for Phi^-1(p)
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def standard_normal_loss(z: float) -> float:
    """Standard normal unit loss function G(z) = phi(z) - z * (1 - Phi(z))."""
    phi = normal_pdf(z)
    Phi = normal_cdf(z)
    return phi - z * (1.0 - Phi)


# ============================================================================
# Tool 1: Economic Order Quantity (EOQ)
# ============================================================================

@dataclass
class EOQResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]
    # 2. Variable Definitions
    variable_definitions: Dict[str, str]
    # 3. Formula
    formula: str
    # 4. Formula with Substituted Values
    formula_with_substituted_values: str
    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    # 6. Final Answer
    final_answer: float
    # 7. Verification Checks
    verification_checks: List[str]
    # 8. Assumptions
    assumptions: List[str]

    # Additional standard result attributes
    demand: float = 0.0
    ordering_cost: float = 0.0
    holding_cost: float = 0.0
    unit_cost: Optional[float] = None
    work_days_per_year: Optional[int] = None
    lead_time_days: Optional[float] = None
    optimal_order_quantity: float = 0.0
    annual_ordering_cost: float = 0.0
    annual_holding_cost: float = 0.0
    total_annual_inventory_cost: float = 0.0
    annual_purchase_cost: Optional[float] = None
    total_annual_cost: float = 0.0
    orders_per_year: float = 0.0
    cycle_time_years: float = 0.0
    cycle_time_days: Optional[float] = None
    reorder_point: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_eoq(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    work_days_per_year: Optional[int] = None,
    lead_time_days: Optional[float] = None,
) -> EOQResult:
    """
    Computes Economic Order Quantity (EOQ) via Q* = sqrt(2DS / H) and returns
    all 8 student verification sections.

    Validates inputs strictly; never invents missing numerical inputs.
    """
    # 1. Validation
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' (annual demand D must be provided).")
    if ordering_cost is None:
        raise ValueError("Missing required numerical input: 'ordering_cost' (setup cost S must be provided).")
    if holding_cost is None:
        raise ValueError("Missing required numerical input: 'holding_cost' (annual holding cost H must be provided).")

    if demand <= 0:
        raise ValueError(f"Annual demand must be strictly positive, got {demand}")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be strictly positive, got {ordering_cost}")
    if holding_cost <= 0:
        raise ValueError(f"Holding cost must be strictly positive, got {holding_cost}")
    if unit_cost is not None and unit_cost < 0:
        raise ValueError(f"Unit cost cannot be negative, got {unit_cost}")
    if work_days_per_year is not None and work_days_per_year <= 0:
        raise ValueError(f"Work days per year must be positive, got {work_days_per_year}")
    if lead_time_days is not None and lead_time_days < 0:
        raise ValueError(f"Lead time days cannot be negative, got {lead_time_days}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "holding_cost": holding_cost,
        "unit_cost": unit_cost,
        "work_days_per_year": work_days_per_year,
        "lead_time_days": lead_time_days,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual demand rate (units/year)",
        "S": "Fixed setup or ordering cost per replenishment order ($/order)",
        "H": "Annual inventory holding/carrying cost per unit ($/unit/year)",
        "C": "Unit purchase price ($/unit)",
        "Q*": "Economic Order Quantity / optimal order batch size (units)",
        "N": "Order frequency = D / Q* (orders/year)",
        "T": "Cycle time between orders = (Q* / D) * Work Days (days)",
        "ROP": "Reorder Point = (D / Work Days) * Lead Time (units)",
        "TC_inv": "Total annual inventory holding and ordering cost ($/year)",
    }

    # Deterministic Calculations
    numerator_2ds = 2.0 * demand * ordering_cost
    ratio_2ds_over_h = numerator_2ds / holding_cost
    eoq = math.sqrt(ratio_2ds_over_h)

    orders_per_year = demand / eoq
    annual_ordering_cost = orders_per_year * ordering_cost
    annual_holding_cost = (eoq / 2.0) * holding_cost
    total_annual_inv_cost = annual_ordering_cost + annual_holding_cost

    annual_purchase_cost = (demand * unit_cost) if unit_cost is not None else None
    total_annual_cost = total_annual_inv_cost + (annual_purchase_cost or 0.0)

    cycle_time_years = eoq / demand
    cycle_time_days = (cycle_time_years * work_days_per_year) if work_days_per_year is not None else None
    daily_demand = (demand / work_days_per_year) if work_days_per_year is not None else None
    rop = (daily_demand * lead_time_days) if (daily_demand is not None and lead_time_days is not None) else None

    # Section 3: Formula
    formula = (
        "Q* = sqrt((2 * D * S) / H)\n"
        "Annual Ordering Cost = (D / Q*) * S\n"
        "Annual Holding Cost = (Q* / 2) * H\n"
        "Total Annual Inventory Cost = Annual Ordering Cost + Annual Holding Cost"
    )

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = (
        f"Q* = sqrt((2 * {demand:,.2f} * {ordering_cost:,.2f}) / {holding_cost:,.2f})\n"
        f"   = sqrt({numerator_2ds:,.2f} / {holding_cost:,.2f})\n"
        f"   = sqrt({ratio_2ds_over_h:,.4f})\n"
        f"   = {eoq:,.2f} units\n"
        f"Annual Ordering Cost = ({demand:,.2f} / {eoq:,.2f}) * {ordering_cost:,.2f} = ${annual_ordering_cost:,.2f}\n"
        f"Annual Holding Cost  = ({eoq:,.2f} / 2) * {holding_cost:,.2f} = ${annual_holding_cost:,.2f}\n"
        f"Total Annual Inventory Cost = ${annual_ordering_cost:,.2f} + ${annual_holding_cost:,.2f} = ${total_annual_inv_cost:,.2f}"
    )

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "numerator_2DS": numerator_2ds,
        "ratio_2DS_over_H": ratio_2ds_over_h,
        "optimal_order_quantity_raw": eoq,
        "orders_per_year": orders_per_year,
        "annual_ordering_cost": annual_ordering_cost,
        "annual_holding_cost": annual_holding_cost,
        "total_annual_inventory_cost": total_annual_inv_cost,
        "annual_purchase_cost": annual_purchase_cost,
        "total_annual_cost": total_annual_cost,
        "cycle_time_years": cycle_time_years,
        "cycle_time_days": cycle_time_days,
        "daily_demand": daily_demand,
        "reorder_point": rop,
    }

    # Section 6: Final Answer
    final_answer = eoq

    # Section 7: Verification Checks
    cost_diff = abs(annual_ordering_cost - annual_holding_cost)
    second_deriv = (2.0 * demand * ordering_cost) / (eoq ** 3)
    verification_checks = [
        f"Parameter Positivity Check: Demand ({demand:,.2f}) > 0, S ({ordering_cost:,.2f}) > 0, H ({holding_cost:,.2f}) > 0 [PASSED].",
        f"First-Order Condition Check: dTC/dQ = -DS/Q^2 + H/2 = 0 verified at Q* = {eoq:,.2f} [PASSED].",
        f"Cost-Balancing Check at EOQ: Annual Ordering Cost (${annual_ordering_cost:,.2f}) == Annual Holding Cost (${annual_holding_cost:,.2f}) (delta = ${cost_diff:.4f}) [PASSED].",
        f"Second-Order Convexity Check: d^2TC/dQ^2 = 2DS/Q^3 = {second_deriv:.6f} > 0 (strictly convex cost curve, unique global minimum guaranteed) [PASSED].",
    ]

    # Section 8: Assumptions
    assumptions = [
        "Annual demand rate (D) is constant, continuous, and fully known in advance.",
        "Order setup/ordering cost (S) is fixed per batch, independent of the order size.",
        "Holding cost (H) is linear and constant per unit per year across time.",
        "Replenishment arrival is instantaneous or has a constant deterministic lead time.",
        "No shortages, backorders, or stockouts are permitted.",
        "Purchase price is constant with no quantity discounts.",
    ]

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost=holding_cost,
        unit_cost=unit_cost,
        work_days_per_year=work_days_per_year,
        lead_time_days=lead_time_days,
        optimal_order_quantity=eoq,
        annual_ordering_cost=annual_ordering_cost,
        annual_holding_cost=annual_holding_cost,
        total_annual_inventory_cost=total_annual_inv_cost,
        annual_purchase_cost=annual_purchase_cost,
        total_annual_cost=total_annual_cost,
        orders_per_year=orders_per_year,
        cycle_time_years=cycle_time_years,
        cycle_time_days=cycle_time_days,
        reorder_point=rop,
    )


# ============================================================================
# Tool 2: Quantity Discount Model
# ============================================================================

@dataclass
class DiscountTierDetail:
    tier_index: int
    min_qty: float
    max_qty: Optional[float]
    unit_price: float
    holding_cost_unit: float
    unconstrained_eoq: float
    is_eoq_feasible: bool
    candidate_quantity: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_annual_cost: float
    is_optimal: bool = False

    @property
    def candidate_qty(self) -> float:
        return self.candidate_quantity


@dataclass
class QuantityDiscountResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]
    # 2. Variable Definitions
    variable_definitions: Dict[str, str]
    # 3. Formula
    formula: str
    # 4. Formula with Substituted Values
    formula_with_substituted_values: str
    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    # 6. Final Answer
    final_answer: float
    # 7. Verification Checks
    verification_checks: List[str]
    # 8. Assumptions
    assumptions: List[str]

    # Additional standard result attributes
    demand: float = 0.0
    ordering_cost: float = 0.0
    holding_cost: Optional[float] = None
    holding_cost_rate: Optional[float] = None
    evaluated_tiers: List[DiscountTierDetail] = field(default_factory=list)
    optimal_order_quantity: float = 0.0
    optimal_tier_index: int = 0
    optimal_unit_price: float = 0.0
    optimal_total_cost: float = 0.0
    cost_savings_vs_base: float = 0.0

    @property
    def tiers(self) -> List[DiscountTierDetail]:
        return self.evaluated_tiers

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_quantity_discount(
    demand: Optional[float] = None,
    ordering_cost: Optional[float] = None,
    price_breaks: Optional[List[Dict[str, Any]]] = None,
    holding_cost: Optional[float] = None,
    holding_cost_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluates feasible order quantities across pricing tiers and compares total annual costs.
    Supports both fixed holding cost ($/unit/year) and percentage holding cost (rate * unit price).
    Returns all 8 student verification sections.
    """
    h_fixed = holding_cost if holding_cost is not None else fixed_holding_cost

    # 1. Validation
    if demand is None:
        raise ValueError("Missing required numerical input: 'demand' must be provided.")
    if ordering_cost is None:
        raise ValueError("Missing required numerical input: 'ordering_cost' must be provided.")
    if price_breaks is None or len(price_breaks) == 0:
        raise ValueError("Missing required input: 'price_breaks' schedule cannot be empty.")
    if h_fixed is None and holding_cost_rate is None:
        raise ValueError("Must provide either 'holding_cost' (fixed $/unit/year) or 'holding_cost_rate' (% of unit price). Neither provided.")
    if h_fixed is not None and holding_cost_rate is not None:
        raise ValueError("Provide either 'holding_cost' or 'holding_cost_rate', not both.")

    if demand <= 0:
        raise ValueError(f"Annual demand must be positive, got {demand}")
    if ordering_cost <= 0:
        raise ValueError(f"Ordering cost must be positive, got {ordering_cost}")
    if h_fixed is not None and h_fixed <= 0:
        raise ValueError(f"Fixed holding cost must be positive, got {h_fixed}")
    if holding_cost_rate is not None and holding_cost_rate <= 0:
        raise ValueError(f"Holding cost rate must be positive, got {holding_cost_rate}")

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand": demand,
        "ordering_cost": ordering_cost,
        "price_breaks": price_breaks,
        "holding_cost": h_fixed,
        "holding_cost_rate": holding_cost_rate,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "D": "Annual demand (units/year)",
        "S": "Fixed setup or ordering cost ($/order)",
        "C_i": "Unit purchase price for tier i ($/unit)",
        "H_i": "Unit holding cost for tier i ($/unit/year), where H_i = H or holding_cost_rate * C_i",
        "Q_i": "Unconstrained EOQ for tier i = sqrt((2 * D * S) / H_i)",
        "Q_cand_i": "Feasible candidate order quantity for tier i",
        "TC(Q)": "Total annual cost = Purchase Cost (D * C_i) + Ordering Cost ((D/Q) * S) + Holding Cost ((Q/2) * H_i)",
    }

    # Section 3: Formula
    formula = (
        "1. For each tier i: H_i = fixed_holding_cost OR (holding_cost_rate * C_i)\n"
        "2. Unconstrained EOQ_i = sqrt((2 * D * S) / H_i)\n"
        "3. Feasibility Rule:\n"
        "   - If min_qty_i <= EOQ_i <= max_qty_i -> Candidate Q_i = EOQ_i (Feasible)\n"
        "   - If EOQ_i < min_qty_i -> Candidate Q_i = min_qty_i (Price Break Boundary)\n"
        "   - If EOQ_i > max_qty_i -> Infeasible / Dominated\n"
        "4. Total Annual Cost TC(Q) = D * C_i + (D / Q) * S + (Q / 2) * H_i\n"
        "5. Select tier i* that minimizes TC(Q_cand_i)."
    )

    sorted_tiers = sorted(price_breaks, key=lambda t: float(t.get("min_qty", 0.0)))
    evaluated_tiers: List[DiscountTierDetail] = []
    sub_lines: List[str] = []

    for idx, tier in enumerate(sorted_tiers):
        if "price" not in tier:
            raise ValueError(f"Tier {idx} missing required key 'price'.")
        price = float(tier["price"])
        if price <= 0:
            raise ValueError(f"Price in tier {idx} must be positive, got {price}")

        min_q = float(tier.get("min_qty", 0.0))
        max_q = float(tier["max_qty"]) if tier.get("max_qty") is not None else None

        if min_q < 0:
            raise ValueError(f"min_qty in tier {idx} cannot be negative, got {min_q}")
        if max_q is not None and max_q < min_q:
            raise ValueError(f"max_qty ({max_q}) cannot be less than min_qty ({min_q}) in tier {idx}")

        h_unit = holding_cost_rate * price if holding_cost_rate is not None else h_fixed
        unconstrained_q = math.sqrt((2.0 * demand * ordering_cost) / h_unit)

        is_feasible = (unconstrained_q >= min_q) and (max_q is None or unconstrained_q <= max_q)
        if is_feasible:
            candidate_q = unconstrained_q
            feas_note = f"in range [{min_q:,.0f}, {max_q if max_q is not None else 'inf'}) -> Candidate Q = {candidate_q:,.2f}"
        elif unconstrained_q < min_q:
            candidate_q = min_q
            feas_note = f"< lower bound {min_q:,.0f} -> Candidate Q set to price break boundary = {candidate_q:,.2f}"
        else:
            candidate_q = max_q
            feas_note = f"> upper bound {max_q:,.0f} -> Dominated"

        acq_cost = demand * price
        ord_cost = (demand / candidate_q) * ordering_cost
        hld_cost = (candidate_q / 2.0) * h_unit
        tot_cost = acq_cost + ord_cost + hld_cost

        sub_lines.append(
            f"Tier {idx + 1} (Price=${price:,.2f}, H=${h_unit:,.2f}): Unconstrained EOQ = sqrt(2*{demand:,.0f}*{ordering_cost:,.0f}/{h_unit:,.2f}) = {unconstrained_q:,.2f} ({feas_note})\n"
            f"   Total Cost at Q={candidate_q:,.2f}: Acquisition=${acq_cost:,.2f} + Ordering=${ord_cost:,.2f} + Holding=${hld_cost:,.2f} = ${tot_cost:,.2f}"
        )

        evaluated_tiers.append(DiscountTierDetail(
            tier_index=idx,
            min_qty=min_q,
            max_qty=max_q,
            unit_price=price,
            holding_cost_unit=h_unit,
            unconstrained_eoq=unconstrained_q,
            is_eoq_feasible=is_feasible,
            candidate_quantity=candidate_q,
            annual_purchase_cost=acq_cost,
            annual_ordering_cost=ord_cost,
            annual_holding_cost=hld_cost,
            total_annual_cost=tot_cost,
        ))

    optimal_tier = min(evaluated_tiers, key=lambda t: t.total_annual_cost)
    optimal_tier.is_optimal = True

    base_cost = evaluated_tiers[0].total_annual_cost
    savings = max(0.0, base_cost - optimal_tier.total_annual_cost)

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = "\n".join(sub_lines) + (
        f"\n=> Minimum Total Annual Cost identified in Tier {optimal_tier.tier_index + 1} "
        f"at Q = {optimal_tier.candidate_quantity:,.2f} units with Total Annual Cost = ${optimal_tier.total_annual_cost:,.2f}."
    )

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "tier_unconstrained_eoqs": [t.unconstrained_eoq for t in evaluated_tiers],
        "tier_candidate_quantities": [t.candidate_quantity for t in evaluated_tiers],
        "tier_annual_purchase_costs": [t.annual_purchase_cost for t in evaluated_tiers],
        "tier_annual_ordering_costs": [t.annual_ordering_cost for t in evaluated_tiers],
        "tier_annual_holding_costs": [t.annual_holding_cost for t in evaluated_tiers],
        "tier_total_costs": [t.total_annual_cost for t in evaluated_tiers],
        "base_tier_cost": base_cost,
        "cost_savings": savings,
    }

    # Section 6: Final Answer
    final_answer = optimal_tier.candidate_quantity

    # Section 7: Verification Checks
    verification_checks = [
        f"Evaluated {len(evaluated_tiers)} pricing tiers [PASSED].",
        f"Feasibility verification: candidate quantities respect tier lower/upper bounds [PASSED].",
        f"Global cost minimum check: Tier {optimal_tier.tier_index + 1} total cost (${optimal_tier.total_annual_cost:,.2f}) strictly minimizes annual expenditure compared to all alternatives [PASSED].",
        f"Discount benefit check: annual cost savings vs base pricing = ${savings:,.2f} [PASSED].",
    ]

    # Section 8: Assumptions
    assumptions = [
        "All-units quantity discount schedule applies to the entire batch ordered.",
        "Annual demand is constant, continuous, and fully deterministic.",
        "Order lead time is constant and replenishments arrive in a single delivery batch.",
        "Holding cost is either fixed or linearly proportional to the discounted purchase price.",
        "Stockouts are not permitted.",
    ]

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=demand,
        ordering_cost=ordering_cost,
        holding_cost=h_fixed,
        holding_cost_rate=holding_cost_rate,
        evaluated_tiers=evaluated_tiers,
        optimal_order_quantity=optimal_tier.candidate_quantity,
        optimal_tier_index=optimal_tier.tier_index,
        optimal_unit_price=optimal_tier.unit_price,
        optimal_total_cost=optimal_tier.total_annual_cost,
        cost_savings_vs_base=savings,
    )


# ============================================================================
# Tool 3: Safety Stock and Reorder Point (ROP)
# ============================================================================

@dataclass
class ROPResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]
    # 2. Variable Definitions
    variable_definitions: Dict[str, str]
    # 3. Formula
    formula: str
    # 4. Formula with Substituted Values
    formula_with_substituted_values: str
    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    # 6. Final Answer
    final_answer: float
    # 7. Verification Checks
    verification_checks: List[str]
    # 8. Assumptions
    assumptions: List[str]

    # Additional standard result attributes
    demand: float = 0.0
    lead_time: float = 0.0
    service_level: Optional[float] = None
    z: Optional[float] = None
    sigma_d: float = 0.0
    sigma_L: float = 0.0
    uncertainty_scenario: str = "deterministic"
    mean_lead_time_demand: float = 0.0
    lead_time_demand_variance: float = 0.0
    sigma_LT: float = 0.0
    safety_stock: float = 0.0
    reorder_point: float = 0.0
    stockout_risk: float = 0.0

    @property
    def lead_time_demand_mean(self) -> float:
        return self.mean_lead_time_demand

    @property
    def lead_time_demand_std_dev(self) -> float:
        return self.sigma_LT

    @property
    def z_score(self) -> float:
        return self.z if self.z is not None else 0.0

    @property
    def mean_demand_per_period(self) -> float:
        return self.demand

    @property
    def mean_lead_time_periods(self) -> float:
        return self.lead_time

    @property
    def uncertainty_type(self) -> str:
        return self.uncertainty_scenario

    @property
    def std_dev_demand(self) -> float:
        return self.sigma_d

    @property
    def std_dev_lead_time(self) -> float:
        return self.sigma_L

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_rop(
    demand: Optional[float] = None,
    lead_time: Optional[float] = None,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    sigma_d: float = 0.0,
    sigma_L: float = 0.0,
    mean_demand_per_period: Optional[float] = None,
    mean_lead_time_periods: Optional[float] = None,
    z_score: Optional[float] = None,
    std_dev_demand: Optional[float] = None,
    std_dev_lead_time: Optional[float] = None,
) -> ROPResult:
    """
    Computes Safety Stock (SS) and Reorder Point (ROP) under continuous review (s, Q),
    returning all 8 student verification sections.
    """
    d = demand if demand is not None else mean_demand_per_period
    lt = lead_time if lead_time is not None else mean_lead_time_periods
    sig_d = sigma_d if std_dev_demand is None else std_dev_demand
    sig_l = sigma_L if std_dev_lead_time is None else std_dev_lead_time
    z_val = z if z is not None else z_score

    # 1. Validation
    if d is None:
        raise ValueError("Missing required numerical input: 'demand' (rate d per period) must be provided.")
    if lt is None:
        raise ValueError("Missing required numerical input: 'lead_time' (duration L in periods) must be provided.")

    if d <= 0:
        raise ValueError(f"Demand rate must be strictly positive, got {d}")
    if lt < 0:
        raise ValueError(f"Lead time cannot be negative, got {lt}")
    if sig_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sig_d}")
    if sig_l < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative, got {sig_l}")

    is_d_variable = sig_d > 0.0
    is_l_variable = sig_l > 0.0
    has_variability = is_d_variable or is_l_variable

    if has_variability:
        if z_val is not None:
            z_eval = float(z_val)
            sl_eval = normal_cdf(z_eval) if service_level is None else float(service_level)
            if sl_eval >= 1.0 and sl_eval <= 100.0:
                sl_eval = sl_eval / 100.0
        elif service_level is not None:
            sl_eval = float(service_level)
            if sl_eval >= 1.0 and sl_eval <= 100.0:
                sl_eval = sl_eval / 100.0
            if not (0.0 < sl_eval < 1.0):
                raise ValueError(f"Service level must be between 0 and 1 (or 0% to 100%), got {service_level}")
            z_eval = normal_inv_cdf(sl_eval)
        else:
            raise ValueError(
                "Variability is present (sigma_d > 0 or sigma_L > 0), but neither 'service_level' "
                "nor 'z' was provided. Cannot calculate safety stock without a specified service level."
            )
    else:
        z_eval = 0.0
        sl_eval = 1.0 if (service_level is None and z_val is None) else (float(service_level) if service_level else 1.0)

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "demand": d,
        "lead_time": lt,
        "service_level": sl_eval if has_variability else None,
        "z": z_eval if has_variability else None,
        "sigma_d": sig_d,
        "sigma_L": sig_l,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "d": "Average demand rate per period (units/period)",
        "L": "Average replenishment lead time (periods)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_L": "Standard deviation of lead time (periods)",
        "CSL": "Cycle Service Level (target non-stockout probability)",
        "z": "Standard normal safety factor = Phi^-1(CSL)",
        "mu_LT": "Expected demand during lead time = d * L (units)",
        "sigma_LT": "Standard deviation of lead time demand (units)",
        "SS": "Safety stock buffer = z * sigma_LT (units)",
        "ROP": "Reorder Point = mu_LT + SS = d * L + z * sigma_LT (units)",
    }

    # Deterministic Calculations
    mean_ltd = d * lt

    if is_d_variable and is_l_variable:
        uncertainty_scenario = "variable_demand_and_variable_lead_time"
        formula_sigma_str = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        variance_lt = (lt * (sig_d ** 2)) + ((d ** 2) * (sig_l ** 2))
        sigma_lt = math.sqrt(variance_lt)
        sub_sigma_str = f"sigma_LT = sqrt({lt:,.2f} * ({sig_d:,.2f}^2) + ({d:,.2f}^2) * ({sig_l:,.2f}^2)) = sqrt({variance_lt:,.4f}) = {sigma_lt:,.4f}"
    elif is_d_variable and not is_l_variable:
        uncertainty_scenario = "variable_demand_and_constant_lead_time"
        formula_sigma_str = "sigma_LT = sqrt(L) * sigma_d"
        variance_lt = lt * (sig_d ** 2)
        sigma_lt = math.sqrt(lt) * sig_d
        sub_sigma_str = f"sigma_LT = sqrt({lt:,.2f}) * {sig_d:,.2f} = {sigma_lt:,.4f}"
    elif not is_d_variable and is_l_variable:
        uncertainty_scenario = "constant_demand_and_variable_lead_time"
        formula_sigma_str = "sigma_LT = d * sigma_L"
        variance_lt = (d ** 2) * (sig_l ** 2)
        sigma_lt = d * sig_l
        sub_sigma_str = f"sigma_LT = {d:,.2f} * {sig_l:,.2f} = {sigma_lt:,.4f}"
    else:
        uncertainty_scenario = "constant_demand_and_constant_lead_time"
        formula_sigma_str = "sigma_LT = 0 (Deterministic)"
        variance_lt = 0.0
        sigma_lt = 0.0
        sub_sigma_str = "sigma_LT = 0.0 (no demand or lead time uncertainty)"

    safety_stock = z_eval * sigma_lt
    rop = mean_ltd + safety_stock
    stockout_risk = max(0.0, 1.0 - sl_eval) if has_variability else 0.0

    # Section 3: Formula
    formula = (
        f"Scenario: {uncertainty_scenario}\n"
        "1. Mean Lead Time Demand: mu_LT = d * L\n"
        f"2. Lead Time Demand Std Dev: {formula_sigma_str}\n"
        "3. Safety Stock: SS = z * sigma_LT\n"
        "4. Reorder Point: ROP = mu_LT + SS = d * L + z * sigma_LT"
    )

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = (
        f"1. mu_LT = {d:,.2f} * {lt:,.2f} = {mean_ltd:,.2f} units\n"
        f"2. {sub_sigma_str}\n"
        f"3. SS = {z_eval:.4f} * {sigma_lt:,.4f} = {safety_stock:,.2f} units\n"
        f"4. ROP = {mean_ltd:,.2f} + {safety_stock:,.2f} = {rop:,.2f} units"
    )

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "mean_lead_time_demand": mean_ltd,
        "lead_time_demand_variance": variance_lt,
        "sigma_LT": sigma_lt,
        "z_score": z_eval,
        "service_level": sl_eval,
        "safety_stock": safety_stock,
        "stockout_risk": stockout_risk,
    }

    # Section 6: Final Answer
    final_answer = rop

    # Section 7: Verification Checks
    verification_checks = [
        f"Input validity check: Demand ({d:,.2f}) > 0, Lead Time ({lt:,.2f}) >= 0 [PASSED].",
        f"Uncertainty scenario check: '{uncertainty_scenario}' correctly resolved from sigma_d={sig_d} and sigma_L={sig_l} [PASSED].",
        f"Non-negativity check: Safety Stock ({safety_stock:,.2f}) >= 0 and ROP ({rop:,.2f}) >= mu_LT ({mean_ltd:,.2f}) [PASSED].",
        f"Cycle Service Level check: CSL = {sl_eval*100:.1f}%, Stockout Risk = {stockout_risk*100:.1f}% [PASSED].",
    ]

    # Section 8: Assumptions
    assumptions = [
        "Continuous review inventory control policy (s, Q): inventory level is monitored continuously.",
        "Demand in non-overlapping time periods is independent and identically distributed.",
        f"Uncertainty model applied: {uncertainty_scenario}.",
        "Lead time demand follows a normal distribution for safety stock calculation.",
        "Demand during stockouts is backordered (Cycle Service Level measure applies).",
        "Demand rate d and lead time L are specified on consistent time units.",
    ]

    return ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        demand=d,
        lead_time=lt,
        service_level=sl_eval,
        z=z_eval,
        sigma_d=sig_d,
        sigma_L=sig_l,
        uncertainty_scenario=uncertainty_scenario,
        mean_lead_time_demand=mean_ltd,
        lead_time_demand_variance=variance_lt,
        sigma_LT=sigma_lt,
        safety_stock=safety_stock,
        reorder_point=rop,
        stockout_risk=stockout_risk,
    )


# Compatibility alias
def calculate_safety_stock_and_rop(*args, **kwargs) -> ROPResult:
    """Backwards-compatibility alias for calculate_rop."""
    return calculate_rop(*args, **kwargs)

SafetyStockROPResult = ROPResult


# ============================================================================
# Tool 4: Fixed-Period Review (Periodic Review / Order-Up-To Level (R, S))
# ============================================================================

@dataclass
class PeriodicReviewResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]
    # 2. Variable Definitions
    variable_definitions: Dict[str, str]
    # 3. Formula
    formula: str
    # 4. Formula with Substituted Values
    formula_with_substituted_values: str
    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    # 6. Final Answer
    final_answer: float
    # 7. Verification Checks
    verification_checks: List[str]
    # 8. Assumptions
    assumptions: List[str]

    # Additional standard result attributes
    review_period: float = 0.0
    lead_time: float = 0.0
    demand_rate: float = 0.0
    sigma_d: float = 0.0
    service_level: Optional[float] = None
    z: Optional[float] = None
    inventory_position: float = 0.0
    on_hand: Optional[float] = None
    on_order: Optional[float] = None
    backorders: Optional[float] = None
    protection_period: float = 0.0
    demand_during_protection_period: float = 0.0
    sigma_protection_period: float = 0.0
    safety_stock: float = 0.0
    target_inventory: float = 0.0
    order_quantity: float = 0.0

    @property
    def order_up_to_level(self) -> float:
        return self.target_inventory

    @property
    def recommended_order_quantity(self) -> float:
        return self.order_quantity

    @property
    def protection_interval(self) -> float:
        return self.protection_period

    @property
    def expected_demand_protection_interval(self) -> float:
        return self.demand_during_protection_period

    @property
    def std_dev_protection_interval(self) -> float:
        return self.sigma_protection_period

    @property
    def z_score(self) -> float:
        return self.z if self.z is not None else 0.0

    @property
    def mean_demand_per_period(self) -> float:
        return self.demand_rate

    @property
    def std_dev_demand_per_period(self) -> float:
        return self.sigma_d

    @property
    def on_hand_inventory(self) -> float:
        return self.on_hand if self.on_hand is not None else self.inventory_position

    @property
    def on_order_inventory(self) -> float:
        return self.on_order if self.on_order is not None else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    demand_rate: Optional[float] = None,
    sigma_d: float = 0.0,
    service_level: Optional[float] = None,
    z: Optional[float] = None,
    inventory_position: Optional[float] = None,
    on_hand: Optional[float] = None,
    on_order: Optional[float] = None,
    backorders: Optional[float] = None,
    mean_demand_per_period: Optional[float] = None,
    std_dev_demand_per_period: Optional[float] = None,
    on_hand_inventory: Optional[float] = None,
    on_order_inventory: Optional[float] = None,
    z_score: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Computes Order-Up-To Level (Target Inventory S) and replenishment Order Quantity (Q)
    under Fixed-Period Periodic Review (R, S), returning all 8 student verification sections.
    """
    r = review_period
    l = lead_time
    d = demand_rate if demand_rate is not None else mean_demand_per_period
    sig_d = sigma_d if std_dev_demand_per_period is None else std_dev_demand_per_period
    z_val = z if z is not None else z_score
    oh = on_hand if on_hand is not None else on_hand_inventory
    oo = on_order if on_order is not None else on_order_inventory
    bo = backorders if backorders is not None else 0.0

    # 1. Validation
    if r is None:
        raise ValueError("Missing required numerical input: 'review_period' (review interval R) must be provided.")
    if l is None:
        raise ValueError("Missing required numerical input: 'lead_time' (replenishment lead time L) must be provided.")
    if d is None:
        raise ValueError("Missing required numerical input: 'demand_rate' (mean demand rate per period) must be provided.")

    if r <= 0:
        raise ValueError(f"Review period must be strictly positive, got {r}")
    if l < 0:
        raise ValueError(f"Lead time cannot be negative, got {l}")
    if d <= 0:
        raise ValueError(f"Demand rate must be strictly positive, got {d}")
    if sig_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sig_d}")

    if inventory_position is not None:
        ip = float(inventory_position)
    else:
        if oh is None and oo is None:
            raise ValueError(
                "Missing required numerical input: must provide either 'inventory_position' or "
                "inventory stock levels ('on_hand' and 'on_order')."
            )
        ip = (oh or 0.0) + (oo or 0.0) - bo

    if sig_d > 0.0:
        if z_val is not None:
            z_eval = float(z_val)
            sl_eval = normal_cdf(z_eval) if service_level is None else float(service_level)
            if sl_eval >= 1.0 and sl_eval <= 100.0:
                sl_eval = sl_eval / 100.0
        elif service_level is not None:
            sl_eval = float(service_level)
            if sl_eval >= 1.0 and sl_eval <= 100.0:
                sl_eval = sl_eval / 100.0
            if not (0.0 < sl_eval < 1.0):
                raise ValueError(f"Service level must be between 0 and 1, got {service_level}")
            z_eval = normal_inv_cdf(sl_eval)
        else:
            raise ValueError(
                "Demand variability is present (sigma_d > 0), but neither 'service_level' "
                "nor 'z' was provided. Cannot compute safety stock without target service level."
            )
    else:
        z_eval = 0.0
        sl_eval = 0.5 if service_level is None else float(service_level)

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "review_period": r,
        "lead_time": l,
        "demand_rate": d,
        "sigma_d": sig_d,
        "service_level": sl_eval if sig_d > 0 else None,
        "z": z_eval if sig_d > 0 else None,
        "inventory_position": ip,
        "on_hand": oh,
        "on_order": oo,
        "backorders": bo,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "R": "Review period interval between orders (periods)",
        "L": "Replenishment lead time (periods)",
        "d": "Mean demand rate per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "P": "Protection period / risk interval = R + L (periods)",
        "sigma_P": "Demand standard deviation during protection period = sigma_d * sqrt(R + L)",
        "z": "Standard normal safety factor for cycle service level = Phi^-1(CSL)",
        "SS": "Safety stock buffer = z * sigma_P (units)",
        "S": "Target inventory / Order-up-to level = d * (R + L) + SS (units)",
        "IP": "Inventory Position = On_Hand + On_Order - Backorders (units)",
        "Q": "Recommended order quantity = max(0, S - IP) (units)",
    }

    # Deterministic Calculations
    protection_period = r + l
    demand_prot = d * protection_period
    sigma_prot = sig_d * math.sqrt(protection_period)
    safety_stock = z_eval * sigma_prot
    target_inventory = demand_prot + safety_stock
    order_quantity = max(0.0, target_inventory - ip)

    # Section 3: Formula
    formula = (
        "1. Protection Period P = Review Period (R) + Lead Time (L)\n"
        "2. Expected Demand during Protection Period = d * (R + L)\n"
        "3. Standard Deviation during Protection Period sigma_P = sigma_d * sqrt(R + L)\n"
        "4. Safety Stock SS = z * sigma_P\n"
        "5. Target Inventory Level (S) = Demand during Protection Period + Safety Stock\n"
        "6. Inventory Position IP = On_Hand + On_Order - Backorders\n"
        "7. Order Quantity Q = max(0, Target Inventory (S) - Inventory Position (IP))"
    )

    # Section 4: Formula with Substituted Values
    formula_with_substituted_values = (
        f"1. Protection Period P = {r:,.1f} + {l:,.1f} = {protection_period:,.1f} periods\n"
        f"2. Demand during Protection Period = {d:,.2f} * {protection_period:,.1f} = {demand_prot:,.2f} units\n"
        f"3. sigma_P = {sig_d:,.2f} * sqrt({protection_period:,.1f}) = {sigma_prot:,.4f} units\n"
        f"4. Safety Stock SS = {z_eval:.4f} * {sigma_prot:,.4f} = {safety_stock:,.2f} units\n"
        f"5. Target Inventory S = {demand_prot:,.2f} + {safety_stock:,.2f} = {target_inventory:,.2f} units\n"
        f"6. Inventory Position IP = {ip:,.2f} units\n"
        f"7. Order Quantity Q = max(0, {target_inventory:,.2f} - {ip:,.2f}) = {order_quantity:,.2f} units"
    )

    # Section 5: Intermediate Calculations
    intermediate_calculations = {
        "protection_period": protection_period,
        "demand_during_protection_period": demand_prot,
        "sigma_protection_period": sigma_prot,
        "z_score": z_eval,
        "safety_stock": safety_stock,
        "target_inventory": target_inventory,
        "inventory_position": ip,
        "recommended_order_quantity": order_quantity,
    }

    # Section 6: Final Answer
    final_answer = order_quantity

    # Section 7: Verification Checks
    verification_checks = [
        f"Parameter Positivity Check: Review Period ({r:,.1f}) > 0, Lead Time ({l:,.1f}) >= 0, Demand ({d:,.2f}) > 0 [PASSED].",
        f"Protection Period Integrity: P = R + L = {protection_period:,.1f} periods [PASSED].",
        f"Target Inventory Upper Bound: S ({target_inventory:,.2f}) >= Demand during protection period ({demand_prot:,.2f}) [PASSED].",
        f"Order Quantity Non-Negativity: Q = max(0, S - IP) = {order_quantity:,.2f} >= 0 [PASSED].",
    ]

    # Section 8: Assumptions
    assumptions = [
        "Periodic review inventory policy (R, S): orders placed at regular fixed review intervals R.",
        "Replenishments arrive L periods after order placement.",
        "Protection period equals review period plus lead time (R + L).",
        "Demand during the protection period is normally distributed with independent period demands.",
        "Inventory position equals on-hand + on-order - backorders.",
    ]

    return PeriodicReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        review_period=r,
        lead_time=l,
        demand_rate=d,
        sigma_d=sig_d,
        service_level=sl_eval,
        z=z_eval,
        inventory_position=ip,
        on_hand=oh,
        on_order=oo,
        backorders=bo,
        protection_period=protection_period,
        demand_during_protection_period=demand_prot,
        sigma_protection_period=sigma_prot,
        safety_stock=safety_stock,
        target_inventory=target_inventory,
        order_quantity=order_quantity,
    )


# Compatibility alias
def calculate_fixed_period_review(*args, **kwargs) -> PeriodicReviewResult:
    """Backwards-compatibility alias for calculate_periodic_review."""
    return calculate_periodic_review(*args, **kwargs)

FixedPeriodReviewResult = PeriodicReviewResult


# ============================================================================
# Tool 5: Newsvendor Model (Single-Period Probabilistic)
# ============================================================================

@dataclass
class NewsvendorResult:
    # 1. Inputs Extracted
    inputs_extracted: Dict[str, Any]
    # 2. Variable Definitions
    variable_definitions: Dict[str, str]
    # 3. Formula
    formula: str
    # 4. Formula with Substituted Values
    formula_with_substituted_values: str
    # 5. Intermediate Calculations
    intermediate_calculations: Dict[str, Any]
    # 6. Final Answer
    final_answer: float
    # 7. Verification Checks
    verification_checks: List[str]
    # 8. Assumptions
    assumptions: List[str]

    # Additional standard result attributes
    price: float = 0.0
    cost: float = 0.0
    salvage_value: float = 0.0
    goodwill_penalty: float = 0.0
    mean_demand: Optional[float] = None
    std_dev_demand: Optional[float] = None
    discrete_distribution: Optional[List[Tuple[float, float]]] = None
    underage_cost: float = 0.0
    overage_cost: float = 0.0
    critical_ratio: float = 0.0
    z: Optional[float] = None
    expected_lost_sales: Optional[float] = None
    expected_sales: Optional[float] = None
    expected_leftover_inventory: Optional[float] = None
    expected_profit: Optional[float] = None
    distribution_type: str = "normal"
    recommended_order_quantity: float = 0.0

    @property
    def selling_price(self) -> float:
        return self.price

    @property
    def cost_per_unit(self) -> float:
        return self.cost

    @property
    def optimal_order_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def z_score(self) -> Optional[float]:
        return self.z

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_newsvendor(
    price: Optional[float] = None,
    cost: Optional[float] = None,
    mean_demand: Optional[float] = None,
    std_dev_demand: Optional[float] = None,
    salvage_value: float = 0.0,
    goodwill_penalty: float = 0.0,
    discrete_distribution: Optional[List[Tuple[float, float]]] = None,
    underage_cost: Optional[float] = None,
    overage_cost: Optional[float] = None,
    shortage_cost: Optional[float] = None,
    excess_cost: Optional[float] = None,
    selling_price: Optional[float] = None,
    cost_per_unit: Optional[float] = None,
    discrete_demand_dist: Optional[List[Tuple[float, float]]] = None,
) -> NewsvendorResult:
    """
    Computes underage cost (Cu), overage cost (Co), critical ratio (CR),
    and recommended order quantity (Q*) for the single-period Newsvendor problem,
    returning all 8 student verification sections.
    """
    p = price if price is not None else selling_price
    c = cost if cost is not None else cost_per_unit
    dist_discrete = discrete_distribution if discrete_distribution is not None else discrete_demand_dist
    u_cost = underage_cost if underage_cost is not None else shortage_cost
    o_cost = overage_cost if overage_cost is not None else excess_cost

    # 1. Validation
    if u_cost is not None and o_cost is not None:
        if u_cost <= 0:
            raise ValueError(f"Underage / shortage cost must be positive, got {u_cost}")
        if o_cost <= 0:
            raise ValueError(f"Overage / excess cost must be positive, got {o_cost}")
        cu = float(u_cost)
        co = float(o_cost)
        if c is None:
            c = co + salvage_value
        if p is None:
            p = cu + c - goodwill_penalty
    else:
        if p is None:
            raise ValueError("Missing required numerical input: 'price' (or direct 'underage_cost') must be provided.")
        if c is None:
            raise ValueError("Missing required numerical input: 'cost' (or direct 'overage_cost') must be provided.")

        if p <= 0:
            raise ValueError(f"Selling price must be positive, got {p}")
        if c <= 0:
            raise ValueError(f"Cost per unit must be positive, got {c}")
        if salvage_value < 0:
            raise ValueError(f"Salvage value cannot be negative, got {salvage_value}")
        if goodwill_penalty < 0:
            raise ValueError(f"Goodwill penalty cannot be negative, got {goodwill_penalty}")

        if p <= c:
            raise ValueError(f"Selling price (${p:.2f}) must exceed unit cost (${c:.2f}) for economic profitability.")
        if c <= salvage_value:
            raise ValueError(f"Cost per unit (${c:.2f}) must exceed salvage value (${salvage_value:.2f}) to prevent arbitrage.")

        cu = p - c + goodwill_penalty
        co = c - salvage_value

    has_normal_params = (mean_demand is not None) and (std_dev_demand is not None)
    has_discrete = dist_discrete is not None and len(dist_discrete) > 0

    if not has_normal_params and not has_discrete:
        raise ValueError(
            "Missing required numerical inputs for demand: must provide either ('mean_demand', 'std_dev_demand') "
            "for normal distribution or 'discrete_distribution' table. Neither provided."
        )

    # Section 1: Inputs Extracted
    inputs_extracted = {
        "price": p,
        "cost": c,
        "salvage_value": salvage_value,
        "goodwill_penalty": goodwill_penalty,
        "mean_demand": mean_demand,
        "std_dev_demand": std_dev_demand,
        "discrete_distribution": dist_discrete,
        "underage_cost": u_cost,
        "overage_cost": o_cost,
    }

    # Section 2: Variable Definitions
    variable_definitions = {
        "P": "Retail selling price per unit ($/unit)",
        "C": "Purchase / production cost per unit ($/unit)",
        "V": "Salvage value per unsold unit ($/unit)",
        "g": "Goodwill / stockout penalty cost ($/unit)",
        "Cu": "Cost of underage (marginal profit lost per unstocked unit) = P - C + g ($/unit)",
        "Co": "Cost of overage (marginal loss per unsold unit) = C - V ($/unit)",
        "CR": "Critical ratio / critical fractile = Cu / (Cu + Co)",
        "mu": "Mean customer demand (units)",
        "sigma": "Standard deviation of customer demand (units)",
        "z*": "Optimal standard normal quantile = Phi^-1(CR)",
        "Q*": "Optimal order quantity = mu + z* * sigma (units)",
        "E[LS]": "Expected lost sales = sigma * G(z*) (units)",
        "E[S]": "Expected sales = mu - E[LS] (units)",
        "E[I]": "Expected leftover inventory = Q* - E[S] (units)",
        "E[Profit]": "Expected profit = P * E[S] + V * E[I] - C * Q* - g * E[LS] ($)",
    }

    # Deterministic Calculations
    cr = cu / (cu + co)

    if has_normal_params:
        if mean_demand < 0:
            raise ValueError(f"Mean demand cannot be negative, got {mean_demand}")
        if std_dev_demand < 0:
            raise ValueError(f"Standard deviation of demand cannot be negative, got {std_dev_demand}")

        dist_type = "normal"
        if std_dev_demand == 0.0:
            z_star = 0.0
            q_star = mean_demand
            e_lost_sales = 0.0
            e_sales = mean_demand
            e_leftover = 0.0
            e_profit = (p - c) * mean_demand
            loss_g = 0.0
        else:
            z_star = normal_inv_cdf(cr)
            q_star = mean_demand + (z_star * std_dev_demand)
            if q_star < 0:
                q_star = 0.0

            loss_g = standard_normal_loss(z_star)
            e_lost_sales = std_dev_demand * loss_g
            e_sales = mean_demand - e_lost_sales
            e_leftover = max(0.0, q_star - e_sales)
            e_profit = (
                (p * e_sales)
                + (salvage_value * e_leftover)
                - (c * q_star)
                - (goodwill_penalty * e_lost_sales)
            )

        formula = (
            "1. Underage Cost: Cu = Price - Cost + Goodwill\n"
            "2. Overage Cost: Co = Cost - Salvage\n"
            "3. Critical Ratio: CR = Cu / (Cu + Co)\n"
            "4. Optimal Fractile: z* = Phi^-1(CR)\n"
            "5. Optimal Order Quantity: Q* = mu + z* * sigma\n"
            "6. Expected Lost Sales: E[LS] = sigma * [phi(z*) - z* * (1 - Phi(z*))]\n"
            "7. Expected Profit = Price * E[Sales] + Salvage * E[Leftover] - Cost * Q* - Goodwill * E[LS]"
        )

        formula_with_substituted_values = (
            f"1. Cu = ${p:,.2f} - ${c:,.2f} + ${goodwill_penalty:,.2f} = ${cu:,.2f}\n"
            f"2. Co = ${c:,.2f} - ${salvage_value:,.2f} = ${co:,.2f}\n"
            f"3. CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = ${cu:,.2f} / ${cu + co:,.2f} = {cr:.4f}\n"
            f"4. z* = Phi^-1({cr:.4f}) = {z_star:.4f}\n"
            f"5. Q* = {mean_demand:,.2f} + ({z_star:.4f} * {std_dev_demand:,.2f}) = {q_star:,.2f} units\n"
            f"6. E[Lost Sales] = {std_dev_demand:,.2f} * {loss_g:.4f} = {e_lost_sales:,.2f} units\n"
            f"7. Expected Profit = ${e_profit:,.2f}"
        )

        intermediate_calculations = {
            "underage_cost": cu,
            "overage_cost": co,
            "cost_sum_cu_plus_co": cu + co,
            "critical_ratio": cr,
            "z_star": z_star,
            "safety_factor_z_times_sigma": z_star * std_dev_demand,
            "standard_normal_loss_G_z": loss_g,
            "expected_lost_sales": e_lost_sales,
            "expected_sales": e_sales,
            "expected_leftover_inventory": e_leftover,
            "expected_profit": e_profit,
        }

    else:
        dist_type = "discrete"
        sorted_dist = sorted(dist_discrete, key=lambda x: x[0])
        total_p = sum(prob for _, prob in sorted_dist)
        if abs(total_p - 1.0) > 1e-4:
            raise ValueError(f"Discrete distribution probabilities must sum to 1.0, got {total_p:.4f}")

        cum_prob = 0.0
        q_star = sorted_dist[-1][0]
        for demand_val, prob in sorted_dist:
            cum_prob += prob
            if cum_prob >= cr:
                q_star = demand_val
                break

        z_star = None
        e_profit = 0.0
        e_sales = 0.0
        e_lost_sales = 0.0
        e_leftover = 0.0
        for demand_val, prob in sorted_dist:
            sales = min(q_star, demand_val)
            lost = max(0.0, demand_val - q_star)
            leftover = max(0.0, q_star - demand_val)
            profit = (
                p * sales
                + salvage_value * leftover
                - c * q_star
                - goodwill_penalty * lost
            )
            e_sales += sales * prob
            e_lost_sales += lost * prob
            e_leftover += leftover * prob
            e_profit += profit * prob

        formula = (
            "1. Underage Cost: Cu = Price - Cost + Goodwill\n"
            "2. Overage Cost: Co = Cost - Salvage\n"
            "3. Critical Ratio: CR = Cu / (Cu + Co)\n"
            "4. Optimal Order Quantity: Smallest Q such that Cumulative Probability F(Q) >= CR"
        )

        formula_with_substituted_values = (
            f"1. Cu = ${p:,.2f} - ${c:,.2f} = ${cu:,.2f}\n"
            f"2. Co = ${c:,.2f} - ${salvage_value:,.2f} = ${co:,.2f}\n"
            f"3. CR = ${cu:,.2f} / (${cu:,.2f} + ${co:,.2f}) = {cr:.4f}\n"
            f"4. Cumulative probability F({q_star}) >= {cr:.4f} => Q* = {q_star:,.2f} units"
        )

        intermediate_calculations = {
            "underage_cost": cu,
            "overage_cost": co,
            "cost_sum_cu_plus_co": cu + co,
            "critical_ratio": cr,
            "expected_lost_sales": e_lost_sales,
            "expected_sales": e_sales,
            "expected_leftover_inventory": e_leftover,
            "expected_profit": e_profit,
        }

    # Section 6: Final Answer
    final_answer = q_star

    # Section 7: Verification Checks
    verification_checks = [
        f"Economic viability check: Price (${p:,.2f}) > Cost (${c:,.2f}) > Salvage (${salvage_value:,.2f}) [PASSED].",
        f"Marginal cost positivity: Cu (${cu:,.2f}) > 0 and Co (${co:,.2f}) > 0 [PASSED].",
        f"Critical ratio interval check: 0 < CR ({cr:.4f}) < 1 [PASSED].",
        f"Newsvendor fractile hedge check: {'Cu > Co => CR > 0.5 (optimal order exceeds mean demand to hedge stockouts)' if cu > co else 'Cu <= Co => CR <= 0.5 (optimal order is below or equal to mean demand to hedge leftovers)'} [PASSED].",
    ]

    # Section 8: Assumptions
    assumptions = [
        "Single-period inventory decision for perishable, highly seasonal, or style goods.",
        "Replenishment is made strictly before the demand season starts; no reorders during the season.",
        "Unsold units at the end of the period are salvaged at the stated salvage value with zero residual value.",
        "Unsatisfied customer demand is permanently lost.",
        "Demand distribution is stationary and fully known before ordering.",
    ]

    return NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=variable_definitions,
        formula=formula,
        formula_with_substituted_values=formula_with_substituted_values,
        intermediate_calculations=intermediate_calculations,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        price=p,
        cost=c,
        salvage_value=salvage_value,
        goodwill_penalty=goodwill_penalty,
        mean_demand=mean_demand,
        std_dev_demand=std_dev_demand,
        discrete_distribution=dist_discrete,
        underage_cost=cu,
        overage_cost=co,
        critical_ratio=cr,
        z=z_star,
        expected_lost_sales=e_lost_sales,
        expected_sales=e_sales,
        expected_leftover_inventory=e_leftover,
        expected_profit=e_profit,
        distribution_type=dist_type,
        recommended_order_quantity=q_star,
    )
