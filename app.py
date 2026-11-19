"""
Supply Chain Analytics - Inventory Decision Agent Web Application
==================================================================
Interactive Streamlit application and dual CLI runner for the Supply Chain
Inventory Decision Agent.

Architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supported Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock & Reorder Point (Continuous Review (s, Q))
4. Fixed-Period Review (Periodic Review / Order-Up-To Level (R, S))
5. Newsvendor Model (Single-Period Probabilistic)

Web Usage:
  streamlit run app.py

CLI Usage:
  python app.py --demo
  python app.py --test
  python app.py --question "Our annual demand is 12000 units, ordering cost is $25, holding cost is $2.40. What is our EOQ?"
"""

import sys
import argparse
import unittest
from typing import List, Dict, Any, Optional

import streamlit as st

from inventory_agent import InventoryDecisionAgent, AgentDecisionResponse


# ============================================================================
# Benchmark Demonstration Scenarios
# ============================================================================

BENCHMARK_DEMOS = [
    {
        "title": "1. Economic Order Quantity (EOQ)",
        "model": "eoq",
        "question": (
            "Our annual demand is 12,000 units, ordering setup cost is $25 per order, "
            "and unit holding cost is $2.40 per unit per year. Unit cost is $10.00. "
            "Lead time is 5 days. What is our optimal EOQ, annual costs, and reorder point?"
        ),
        "description": "Continuous demand, fixed order setup cost, and linear holding cost under constant lead time.",
    },
    {
        "title": "2. Quantity Discount Model",
        "model": "quantity_discount",
        "question": (
            "We have an annual demand of 10,000 units with an order cost of $100 per order. "
            "Our annual holding cost rate is 20% of unit price. The supplier provides volume discount "
            "price breaks: 0 to 499 at $10.00, 500 to 999 at $9.80, and 1000+ at $9.60. "
            "Which quantity discount tier and order size minimizes total annual cost?"
        ),
        "description": "Multiple pricing tiers with percentage holding cost and order quantity break-point evaluation.",
    },
    {
        "title": "3. Safety Stock & Reorder Point (Continuous Review)",
        "model": "safety_stock_rop",
        "question": (
            "Under continuous review (s, Q), our average daily demand is 100 units with a "
            "standard deviation of 10 units. Supplier lead time is 4 days with no lead time variability. "
            "We require a 95% cycle service level. What should our safety stock and reorder point (ROP) be?"
        ),
        "description": "Continuous review (s, Q) policy with normal demand uncertainty during replenishment lead time.",
    },
    {
        "title": "4. Fixed-Period Review (Periodic Review / Order-Up-To Level)",
        "model": "fixed_period_review",
        "question": (
            "We manage retail inventory on a periodic review policy. Orders are placed every 14 days, "
            "with a lead time of 7 days. Average daily demand is 20 units with a standard deviation of 4 units. "
            "Target cycle service level is 95%. Currently, we have 80 units on hand and 40 units on order. "
            "What is our target order-up-to level (S) and how many units should we order right now?"
        ),
        "description": "Periodic review (R, S) policy with protection interval (R + L) and inventory position netting.",
    },
    {
        "title": "5. Newsvendor Model (Single-Period Probabilistic)",
        "model": "newsvendor",
        "question": (
            "For an upcoming seasonal fashion catalog, the retail selling price is $100 per unit, "
            "the wholesale cost is $40 per unit, and unsold items can be liquidated at a salvage value of $10 per unit. "
            "Customer demand follows a normal distribution with a mean of 1,000 units and standard deviation of 200 units. "
            "What is our critical fractile, optimal order quantity, and expected profit?"
        ),
        "description": "Single-period perishable goods inventory with underage/overage trade-offs and critical ratio.",
    },
]


# ============================================================================
# Cached Agent Instance
# ============================================================================

@st.cache_resource
def get_inventory_agent() -> InventoryDecisionAgent:
    """Instantiates and caches the Supply Chain Inventory Decision Agent."""
    return InventoryDecisionAgent()


# ============================================================================
# Streamlit Web Application Interface
# ============================================================================

def run_streamlit_app():
    """Renders the interactive Streamlit web application."""
    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = get_inventory_agent()

    # Session State for User Question
    if "user_question" not in st.session_state:
        st.session_state.user_question = BENCHMARK_DEMOS[0]["question"]

    # ------------------------------------------------------------------------
    # Sidebar: Model Overview & Benchmark Demonstration Scenarios
    # ------------------------------------------------------------------------
    with st.sidebar:
        st.header("📦 Inventory Decision Agent")
        st.caption("Agentic Decision Architecture:")
        st.markdown(
            "1. **Question** $\\to$ 2. **Identify Model** $\\to$ 3. **Extract Inputs** $\\to$ "
            "4. **Select Tool** $\\to$ 5. **Calculate** $\\to$ 6. **Verify** $\\to$ 7. **Explain**"
        )
        st.divider()

        st.subheader("🎯 Benchmark Scenarios")
        st.caption("Click any scenario to populate the question box:")

        for idx, demo in enumerate(BENCHMARK_DEMOS, 1):
            if st.button(demo["title"], key=f"demo_btn_{idx}", use_container_width=True):
                st.session_state.user_question = demo["question"]
                st.rerun()

        st.divider()
        st.subheader("Supported OR Models")
        st.markdown(
            "- **EOQ**: Economic Order Quantity\n"
            "- **Quantity Discount**: All-units / break tiers\n"
            "- **Safety Stock & ROP**: Continuous $(s, Q)$\n"
            "- **Periodic Review**: Order-up-to $(R, S)$\n"
            "- **Newsvendor**: Single-period probabilistic"
        )
        st.divider()
        if st.button("Clear / Reset Question", use_container_width=True):
            st.session_state.user_question = ""
            st.rerun()

    # ------------------------------------------------------------------------
    # Main Application Header
    # ------------------------------------------------------------------------
    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "Enter natural-language inventory questions below. The agent parses domain parameters, selects "
        "the deterministic Python mathematical tool, performs verification checks, and outputs student step-by-step verification."
    )

    # ------------------------------------------------------------------------
    # User Input Area
    # ------------------------------------------------------------------------
    input_col, btn_col = st.columns([5, 1])

    with input_col:
        question_input = st.text_area(
            "Inventory Question / Inquiry:",
            value=st.session_state.user_question,
            height=120,
            placeholder="e.g., Our annual demand is 12,000 units, ordering setup cost is $25 per order, and unit holding cost is $2.40 per unit per year...",
            key="question_text_area",
        )

    with btn_col:
        st.write("")
        st.write("")
        process_button = st.button("🚀 Process Question", type="primary", use_container_width=True)

    # Trigger processing on button click or if input is provided
    if process_button:
        st.session_state.user_question = question_input

    active_question = st.session_state.user_question.strip()

    if not active_question:
        st.info("💡 Enter a question above or select one of the 5 benchmark scenarios from the sidebar to begin.")
        return

    # ------------------------------------------------------------------------
    # Agent Execution Pipeline
    # ------------------------------------------------------------------------
    with st.spinner("Executing 7-step decision pipeline..."):
        try:
            response: AgentDecisionResponse = agent.process_question(active_question)
        except Exception as e:
            st.error(f"❌ Error processing question: {str(e)}")
            return

    # ------------------------------------------------------------------------
    # Case A: Incomplete Information (Missing Required Parameters)
    # ------------------------------------------------------------------------
    if not response.is_complete:
        st.warning("⚠️ Additional Information Required")

        col1, col2 = st.columns(2)
        with col1:
            st.info(
                f"**Identified Model:** {response.model_name_display}  \n"
                f"**Confidence:** {response.confidence_score * 100:.1f}%  \n"
                f"**Target Tool:** `inventory_tools.{response.selected_tool_name}`"
            )
        with col2:
            st.error(
                "**Missing Required Parameters:**  \n"
                + "\n".join([f"- **{m}**" for m in response.missing_parameters])
                + "\n\n*(The agent will never invent missing numerical values)*"
            )

        if response.extracted_inputs:
            with st.expander("Extracted Parameters Found in Question", expanded=True):
                for k, v in response.extracted_inputs.items():
                    if k != "price_breaks":
                        st.markdown(f"- **{k}**: `{v}`")
                    else:
                        st.markdown(f"- **price_breaks**: `{v}`")

        st.markdown(response.explanation)
        return

    # ------------------------------------------------------------------------
    # Case B: Complete Information - Render Full Verification & Decisions
    # ------------------------------------------------------------------------
    calc = response.calculation_result

    # Pipeline Status Banner
    st.success(
        f"✅ **Pipeline Executed:** Model Identified as **{response.model_name_display}** "
        f"({response.confidence_score * 100:.1f}% confidence) $\\rightarrow$ "
        f"Dispatched Tool: `inventory_tools.{response.selected_tool_name}`"
    )

    # Tabs for Organized Student Presentation
    tab_audit, tab_exec, tab_raw = st.tabs([
        "🎓 Student Step-by-Step Verification (8 Sections)",
        "📊 Executive Recommendation & Insights",
        "📄 Full Formatted Terminal Report",
    ])

    # ------------------------------------------------------------------------
    # Tab 1: Student Verification (8 Mandatory Sections)
    # ------------------------------------------------------------------------
    with tab_audit:
        st.subheader("Student Step-by-Step Verification Audit")
        st.caption("All intermediate numerical calculations are derived deterministically in Python.")

        # Final Answer Callout Card
        final_val = getattr(calc, "final_answer", None)
        if isinstance(final_val, float):
            st.metric(
                label=f"Final Answer ({response.model_name_display})",
                value=f"{final_val:,.2f}",
                delta=f"Exact: {final_val:,.4f}",
            )
        else:
            st.metric(
                label=f"Final Answer ({response.model_name_display})",
                value=str(final_val),
            )

        col_left, col_right = st.columns(2)

        with col_left:
            # 1. Inputs Extracted
            st.markdown("#### 1. Inputs Extracted")
            if hasattr(calc, "inputs_extracted") and calc.inputs_extracted:
                for k, v in calc.inputs_extracted.items():
                    if v is not None:
                        if isinstance(v, float):
                            val_str = f"{v:,.4f}" if (0 < abs(v) < 1) else f"{v:,.2f}"
                            st.markdown(f"- **{k}**: `{val_str}`")
                        elif isinstance(v, list):
                            st.markdown(f"- **{k}**: `{v}`")
                        else:
                            st.markdown(f"- **{k}**: `{v}`")
            else:
                st.markdown("*(None)*")

            # 2. Variable Definitions
            st.markdown("#### 2. Variable Definitions")
            if hasattr(calc, "variable_definitions") and calc.variable_definitions:
                for var_sym, var_desc in calc.variable_definitions.items():
                    st.markdown(f"- **{var_sym}**: {var_desc}")
            else:
                st.markdown("*(None)*")

        with col_right:
            # 5. Intermediate Calculations
            st.markdown("#### 5. Intermediate Calculations")
            if hasattr(calc, "intermediate_calculations") and calc.intermediate_calculations:
                for ik, iv in calc.intermediate_calculations.items():
                    if isinstance(iv, float):
                        val_str = f"{iv:,.2f}" if (abs(iv) >= 1000 or (0.01 <= abs(iv) < 1000 and abs(iv - round(iv, 2)) < 1e-4)) else f"{iv:,.4f}"
                        st.markdown(f"- **{ik}**: `{val_str}`")
                    elif isinstance(iv, (list, dict)):
                        st.markdown(f"- **{ik}**: `{iv}`")
                    else:
                        st.markdown(f"- **{ik}**: `{iv}`")
            else:
                st.markdown("*(None)*")

            # 6. Final Answer
            st.markdown("#### 6. Final Answer")
            st.markdown(f"**Recommended Optimal Value**: `{final_val}`")

        st.divider()

        # 3. Formula
        st.markdown("#### 3. Formula")
        formula_text = getattr(calc, "formula", "N/A")
        st.code(formula_text, language="text")

        # 4. Formula with Substituted Values
        st.markdown("#### 4. Formula with Substituted Values")
        sub_text = getattr(calc, "formula_with_substituted_values", "N/A")
        st.code(sub_text, language="text")

        col_chk, col_asm = st.columns(2)

        with col_chk:
            # 7. Verification Checks
            st.markdown("#### 7. Verification Checks")
            if hasattr(calc, "verification_checks") and calc.verification_checks:
                for chk in calc.verification_checks:
                    st.success(f"✔ {chk}")
            for w in response.verification.warnings:
                st.warning(f"⚠ {w}")
            for e in response.verification.errors:
                st.error(f"❌ {e}")

        with col_asm:
            # 8. Assumptions
            st.markdown("#### 8. Model Assumptions")
            if hasattr(calc, "assumptions") and calc.assumptions:
                for asm in calc.assumptions:
                    st.markdown(f"- {asm}")

    # ------------------------------------------------------------------------
    # Tab 2: Executive Explanation & Insights
    # ------------------------------------------------------------------------
    with tab_exec:
        st.subheader("Executive Decision & Supply Chain Insights")
        st.markdown(response.explanation)

    # ------------------------------------------------------------------------
    # Tab 3: Complete Raw Terminal Report (Easy for Copy/Pasting)
    # ------------------------------------------------------------------------
    with tab_raw:
        st.subheader("Raw Terminal Report")
        st.caption("Complete formatted text report with all 7 pipeline steps:")
        st.code(response.formatted_output, language="text")


# ============================================================================
# CLI Runner for Terminal Demonstrations and Automated Tests
# ============================================================================

def run_cli_mode():
    """Provides CLI access when app.py is invoked with command line flags."""
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent (CLI Mode)"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Natural-language inventory question to evaluate directly",
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run all 5 inventory model showcase demonstrations in terminal",
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Execute unit tests for all mathematical tools and agent pipeline",
    )

    args = parser.parse_args()
    agent = InventoryDecisionAgent()

    if args.test:
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.demo:
        print(">>> EXECUTING SHOWCASE: ALL 5 INVENTORY MODELS\n")
        for idx, demo in enumerate(BENCHMARK_DEMOS, 1):
            print(f"[{idx}/5] SCENARIO: {demo['title']}")
            response = agent.process_question(demo["question"])
            print(response.formatted_output)
        return

    if args.question:
        response = agent.process_question(args.question)
        print(response.formatted_output)
        return


# ============================================================================
# Application Entrypoint
# ============================================================================

if st.runtime.exists():
    run_streamlit_app()
elif __name__ == "__main__":
    # If specific CLI flags are supplied, execute in CLI mode; otherwise run Streamlit
    cli_flags = {"--demo", "-d", "--test", "-t", "--question", "-q"}
    if any(flag in sys.argv for flag in cli_flags):
        run_cli_mode()
    else:
        run_streamlit_app()
