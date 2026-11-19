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
import pandas as pd
import numpy as np

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

        # --------------------------------------------------------------------
        # Data Upload Section (Last Section in Sidebar)
        # --------------------------------------------------------------------
        st.divider()
        st.header("Data Upload")
        uploaded_csv = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Upload an inventory CSV file containing Product, Category, Units, and Unit Cost.",
        )

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
    else:
        # --------------------------------------------------------------------
        # Agent Execution Pipeline
        # --------------------------------------------------------------------
        with st.spinner("Executing 7-step decision pipeline..."):
            try:
                response: AgentDecisionResponse = agent.process_question(active_question)
            except Exception as e:
                st.error(f"❌ Error processing question: {str(e)}")
                response = None

        if response is not None:
            # Case A: Incomplete Information (Missing Required Parameters)
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
            else:
                # Case B: Complete Information - Render Full Verification & Decisions
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

                # ----------------------------------------------------------------
                # Tab 1: Student Verification (8 Mandatory Sections)
                # ----------------------------------------------------------------
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

                # ----------------------------------------------------------------
                # Tab 2: Executive Explanation & Insights
                # ----------------------------------------------------------------
                with tab_exec:
                    st.subheader("Executive Decision & Supply Chain Insights")
                    st.markdown(response.explanation)

                # ----------------------------------------------------------------
                # Tab 3: Complete Raw Terminal Report (Easy for Copy/Pasting)
                # ----------------------------------------------------------------
                with tab_raw:
                    st.subheader("Raw Terminal Report")
                    st.caption("Complete formatted text report with all 7 pipeline steps:")
                    st.code(response.formatted_output, language="text")

    # ------------------------------------------------------------------------
    # Data Processing & Analysis (When CSV is uploaded)
    # ------------------------------------------------------------------------
    if uploaded_csv is not None:
        render_csv_analysis_section(uploaded_csv)


# ============================================================================
# CSV Data Processing & Analysis (pandas)
# ============================================================================

def perform_multicriteria_classification(
    df: pd.DataFrame,
    sales_col: str,
    hits_col: str,
    pareto_cutoff: float = 0.80,
) -> pd.DataFrame:
    """
    Performs multicriteria inventory classification using Sales and Hits into A, B, C, and D classes.

    Methodology:
      - Bivariate Pareto / 2x2 Matrix Classification on Monetary Impact (Sales) and Transaction Velocity (Hits).
      - Class A: High Sales & High Hits (Critical high-value, high-velocity items)
      - Class B: High Sales & Low Hits  (High-value, slow-moving / lumpy demand items)
      - Class C: Low Sales & High Hits  (Low-value, high-velocity / frequent-pick items)
      - Class D: Low Sales & Low Hits   (Low-value, slow-moving tail / candidates for phase-out)
    """
    df_calc = df.copy()

    # Clean and parse Sales to numeric
    if pd.api.types.is_numeric_dtype(df_calc[sales_col]):
        df_calc["Clean_Sales"] = pd.to_numeric(df_calc[sales_col], errors="coerce").fillna(0.0)
    else:
        df_calc["Clean_Sales"] = pd.to_numeric(
            df_calc[sales_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(0.0)

    # Clean and parse Hits to numeric
    if pd.api.types.is_numeric_dtype(df_calc[hits_col]):
        df_calc["Clean_Hits"] = pd.to_numeric(df_calc[hits_col], errors="coerce").fillna(0.0)
    else:
        df_calc["Clean_Hits"] = pd.to_numeric(
            df_calc[hits_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce",
        ).fillna(0.0)

    total_sales = df_calc["Clean_Sales"].sum()
    total_hits = df_calc["Clean_Hits"].sum()

    # Share percentages
    df_calc["Sales Share %"] = (df_calc["Clean_Sales"] / total_sales * 100.0) if total_sales > 0 else 0.0
    df_calc["Hits Share %"] = (df_calc["Clean_Hits"] / total_hits * 100.0) if total_hits > 0 else 0.0

    # Pareto Ranking for Sales ($)
    sorted_sales = df_calc.sort_values(by="Clean_Sales", ascending=False)
    if total_sales > 0:
        prev_cum_sales = (sorted_sales["Clean_Sales"].cumsum() - sorted_sales["Clean_Sales"]) / total_sales
        sales_high_indices = sorted_sales[prev_cum_sales < pareto_cutoff].index
    else:
        sales_high_indices = pd.Index([])

    # Pareto Ranking for Hits
    sorted_hits = df_calc.sort_values(by="Clean_Hits", ascending=False)
    if total_hits > 0:
        prev_cum_hits = (sorted_hits["Clean_Hits"].cumsum() - sorted_hits["Clean_Hits"]) / total_hits
        hits_high_indices = sorted_hits[prev_cum_hits < pareto_cutoff].index
    else:
        hits_high_indices = pd.Index([])

    # Velocity assignment
    df_calc["Sales Velocity"] = np.where(df_calc.index.isin(sales_high_indices), "High", "Low")
    df_calc["Hits Velocity"] = np.where(df_calc.index.isin(hits_high_indices), "High", "Low")

    # Multicriteria classification into A, B, C, D
    def assign_class(row):
        is_sh = row["Sales Velocity"] == "High"
        is_hh = row["Hits Velocity"] == "High"
        if is_sh and is_hh:
            return "A"
        elif is_sh and not is_hh:
            return "B"
        elif not is_sh and is_hh:
            return "C"
        else:
            return "D"

    df_calc["Inventory Class"] = df_calc.apply(assign_class, axis=1)

    descriptions = {
        "A": "Class A: High Sales, High Hits (Fast Movers, High Value)",
        "B": "Class B: High Sales, Low Hits (Slow Movers, High Value)",
        "C": "Class C: Low Sales, High Hits (Fast Movers, Low Value)",
        "D": "Class D: Low Sales, Low Hits (Slow Movers, Low Value)",
    }
    df_calc["Class Description"] = df_calc["Inventory Class"].map(descriptions)

    return df_calc


def render_multicriteria_classification_section(df: pd.DataFrame, sales_col: str, hits_col: str):
    """Renders the multicriteria inventory classification dashboard in Streamlit."""
    st.subheader("🏷️ Multicriteria Inventory Classification (Classes A, B, C, D)")
    st.caption(
        "Multicriteria cross-classification on Monetary Value (**Sales**) and Transaction Frequency (**Hits**) "
        "using bivariate Pareto analysis."
    )

    # Classify the dataframe
    df_classified = perform_multicriteria_classification(df, sales_col, hits_col, pareto_cutoff=0.80)

    total_items = len(df_classified)
    total_sales = df_classified["Clean_Sales"].sum()
    total_hits = df_classified["Clean_Hits"].sum()
    avg_sales = df_classified["Clean_Sales"].mean() if total_items > 0 else 0.0

    # Dataset high-level overview metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Items", f"{total_items:,}")
    with c2:
        st.metric("Total Sales", f"${total_sales:,.2f}")
    with c3:
        st.metric("Total Hits", f"{total_hits:,.0f}")
    with c4:
        st.metric("Avg Sales / Item", f"${avg_sales:,.2f}")

    st.markdown("---")

    # Multicriteria Quadrant Breakdown Cards
    st.markdown("#### Multicriteria Class Distribution")
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)

    classes_count = df_classified["Inventory Class"].value_counts().to_dict()
    classes_sales = df_classified.groupby("Inventory Class")["Clean_Sales"].sum().to_dict()

    with q_col1:
        cnt_a = classes_count.get("A", 0)
        pct_a = (cnt_a / total_items * 100.0) if total_items > 0 else 0.0
        sales_a = classes_sales.get("A", 0.0)
        st.success(f"**Class A** (High Sales, High Hits)\n\n**{cnt_a} items** ({pct_a:.1f}%)\n\nSales: ${sales_a:,.2f}")

    with q_col2:
        cnt_b = classes_count.get("B", 0)
        pct_b = (cnt_b / total_items * 100.0) if total_items > 0 else 0.0
        sales_b = classes_sales.get("B", 0.0)
        st.info(f"**Class B** (High Sales, Low Hits)\n\n**{cnt_b} items** ({pct_b:.1f}%)\n\nSales: ${sales_b:,.2f}")

    with q_col3:
        cnt_c = classes_count.get("C", 0)
        pct_c = (cnt_c / total_items * 100.0) if total_items > 0 else 0.0
        sales_c = classes_sales.get("C", 0.0)
        st.warning(f"**Class C** (Low Sales, High Hits)\n\n**{cnt_c} items** ({pct_c:.1f}%)\n\nSales: ${sales_c:,.2f}")

    with q_col4:
        cnt_d = classes_count.get("D", 0)
        pct_d = (cnt_d / total_items * 100.0) if total_items > 0 else 0.0
        sales_d = classes_sales.get("D", 0.0)
        st.error(f"**Class D** (Low Sales, Low Hits)\n\n**{cnt_d} items** ({pct_d:.1f}%)\n\nSales: ${sales_d:,.2f}")

    # Aggregated Summary Table
    st.markdown("#### Summary Table by Inventory Class")
    summary = (
        df_classified.groupby("Inventory Class", observed=False)
        .agg(
            Item_Count=(df_classified.columns[0], "count"),
            Total_Sales=("Clean_Sales", "sum"),
            Total_Hits=("Clean_Hits", "sum"),
        )
        .reindex(["A", "B", "C", "D"], fill_value=0)
        .reset_index()
    )
    summary["% of Items"] = (summary["Item_Count"] / total_items * 100.0) if total_items > 0 else 0.0
    summary["% of Sales"] = (summary["Total_Sales"] / total_sales * 100.0) if total_sales > 0 else 0.0
    summary["% of Hits"] = (summary["Total_Hits"] / total_hits * 100.0) if total_hits > 0 else 0.0

    policy_map = {
        "A": "Tight continuous review (s, Q), high target service level (98%+), frequent small replenishment.",
        "B": "Strict holding-cost control, lumpy demand management, make-to-order or lean safety stock.",
        "C": "Periodic review (R, S) / visual two-bin Kanban, high safety buffer (cheap to stock, frequent pick).",
        "D": "Rationalize SKU, minimum order quantity, check for obsolescence / dead stock.",
    }
    summary["Strategic Inventory Policy"] = summary["Inventory Class"].map(policy_map)

    st.dataframe(
        summary.style.format({
            "Item_Count": "{:,}",
            "% of Items": "{:.1f}%",
            "Total_Sales": "${:,.2f}",
            "% of Sales": "{:.1f}%",
            "Total_Hits": "{:,.0f}",
            "% of Hits": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Classified Dataframe Display
    st.markdown("#### Classified Inventory Dataframe (with A, B, C, D Classes)")

    # Optional filter by class
    f_col1, f_col2 = st.columns([1, 3])
    with f_col1:
        selected_filter = st.selectbox(
            "Filter by Class:",
            options=["All Classes", "Class A", "Class B", "Class C", "Class D"],
            key="class_filter_select",
        )
    with f_col2:
        search_query = st.text_input("Search by Item Name or Number:", placeholder="e.g., Widget, 1001...", key="item_search_input")

    filtered_df = df_classified.copy()
    if selected_filter != "All Classes":
        target_class = selected_filter.replace("Class ", "")
        filtered_df = filtered_df[filtered_df["Inventory Class"] == target_class]
    if search_query:
        mask = (
            filtered_df["Item Number"].astype(str).str.contains(search_query, case=False, na=False)
            | filtered_df["Item Name"].astype(str).str.contains(search_query, case=False, na=False)
        )
        filtered_df = filtered_df[mask]

    # Present formatted display columns
    display_cols = [
        "Item Number",
        "Item Name",
        "UoM",
        sales_col,
        hits_col,
        "Sales Share %",
        "Hits Share %",
        "Sales Velocity",
        "Hits Velocity",
        "Inventory Class",
        "Class Description",
    ]
    # Filter to existing columns in dataframe
    valid_display_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[valid_display_cols].style.format({
            sales_col: "${:,.2f}" if filtered_df[sales_col].dtype != object else "{}",
            hits_col: "{:,.0f}" if filtered_df[hits_col].dtype != object else "{}",
            "Sales Share %": "{:.2f}%",
            "Hits Share %": "{:.2f}%",
        }),
        use_container_width=True,
    )

    with st.expander("View Raw Uploaded Dataframe", expanded=False):
        st.dataframe(df, use_container_width=True)


def render_units_analysis_section(df: pd.DataFrame):
    """Renders the standard Product, Category, Units, and Unit Cost inventory analysis."""
    st.subheader("📦 Inventory Units & Category Analysis")

    # Display dataset metadata
    col_r, col_c, col_cols = st.columns([1, 1, 3])
    with col_r:
        st.metric("Number of Rows", f"{len(df):,}")
    with col_c:
        st.metric("Number of Columns", f"{len(df.columns):,}")
    with col_cols:
        st.info(f"**Column Names ({len(df.columns)}):** {', '.join(df.columns.tolist())}")

    # Prepare numeric columns for calculation
    df_calc = df.copy()
    if pd.api.types.is_numeric_dtype(df_calc["Units"]):
        df_calc["Units"] = pd.to_numeric(df_calc["Units"], errors="coerce").fillna(0.0)
    else:
        df_calc["Units"] = pd.to_numeric(
            df_calc["Units"].astype(str).str.replace(",", "").str.strip(),
            errors="coerce",
        ).fillna(0.0)

    if pd.api.types.is_numeric_dtype(df_calc["Unit Cost"]):
        df_calc["Unit Cost"] = pd.to_numeric(df_calc["Unit Cost"], errors="coerce").fillna(0.0)
    else:
        df_calc["Unit Cost"] = pd.to_numeric(
            df_calc["Unit Cost"].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.strip(),
            errors="coerce",
        ).fillna(0.0)

    # 1. Total Units
    total_units = df_calc["Units"].sum()

    # 2. Average Unit Cost
    avg_unit_cost = df_calc["Unit Cost"].mean()

    # 3. Identify Product with highest Units
    if not df_calc["Units"].dropna().empty:
        max_idx = df_calc["Units"].idxmax()
        highest_units_product = df_calc.loc[max_idx, "Product"]
        highest_units_count = df_calc.loc[max_idx, "Units"]
    else:
        highest_units_product = "N/A"
        highest_units_count = 0.0

    # 4. Group data by Category and calculate total Units
    category_summary = (
        df_calc.groupby("Category")["Units"]
        .sum()
        .reset_index()
        .rename(columns={"Units": "Total Units"})
        .sort_values(by="Total Units", ascending=False)
    )

    # 5. Calculated column: Inventory Value = Units * Unit Cost
    df_calc["Inventory Value"] = df_calc["Units"] * df_calc["Unit Cost"]

    # 6. Total Inventory Value across all products
    total_inventory_value = df_calc["Inventory Value"].sum()

    # Summary Metrics
    st.markdown("#### Key Inventory Metrics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Units", f"{total_units:,.0f}")
    with m2:
        st.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
    with m3:
        st.metric(
            "Product with Highest Units",
            str(highest_units_product),
            delta=f"{highest_units_count:,.0f} units",
        )
    with m4:
        st.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

    # Total Units by Category
    st.markdown("#### Total Units by Category")
    st.dataframe(
        category_summary.style.format({"Total Units": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

    # Dataframe including the new Inventory Value column
    st.markdown("#### Inventory Dataframe (with Calculated Inventory Value)")
    st.dataframe(
        df_calc.style.format({
            "Units": "{:,.0f}",
            "Unit Cost": "${:,.2f}",
            "Inventory Value": "${:,.2f}",
        }),
        use_container_width=True,
    )

    with st.expander("View Raw Uploaded Dataframe", expanded=False):
        st.dataframe(df, use_container_width=True)


def render_csv_analysis_section(uploaded_csv):
    """
    Processes an uploaded CSV file, detects the schema format, and runs either
    Multicriteria Inventory Classification or Product Units/Value Analysis.
    """
    st.divider()
    st.header("📊 Uploaded Inventory Data Analysis")

    # 1. Read CSV into a pandas dataframe
    try:
        df = pd.read_csv(uploaded_csv)
    except Exception as e:
        st.error(f"❌ Failed to parse uploaded CSV file: {str(e)}")
        return

    # Schema Detection:
    # Format A: Multicriteria Classification ("Item Number", "Item Name", "UoM", "Sales ($)" / "Sales", "Hits")
    sales_col = next((c for c in df.columns if c in ["Sales ($)", "Sales", "Sales($)"] or "sales" in c.lower()), None)
    hits_col = next((c for c in df.columns if c.strip().lower() == "hits" or "hits" in c.lower()), None)
    has_classification_core = all(c in df.columns for c in ["Item Number", "Item Name", "UoM"])
    is_classification_format = has_classification_core and (sales_col is not None) and (hits_col is not None)

    # Format B: Product Units Analysis ("Product", "Category", "Units", "Unit Cost")
    units_required = ["Product", "Category", "Units", "Unit Cost"]
    is_units_format = all(c in df.columns for c in units_required)

    if not is_classification_format and not is_units_format:
        # Check specific missing columns for classification format if partially matching
        st.error("❌ The uploaded CSV does not match a supported inventory schema format.")
        col_help1, col_help2 = st.columns(2)
        with col_help1:
            st.info(
                "**Schema 1: Multicriteria Inventory Classification (A, B, C, D)** requires columns:\n"
                "- `Item Number`\n- `Item Name`\n- `UoM`\n- `Sales ($)` (or `Sales`)\n- `Hits`"
            )
        with col_help2:
            st.info(
                "**Schema 2: Product & Inventory Value Analysis** requires columns:\n"
                "- `Product`\n- `Category`\n- `Units`\n- `Unit Cost`"
            )
        return

    # Execute Multicriteria Classification if applicable
    if is_classification_format:
        render_multicriteria_classification_section(df, sales_col, hits_col)

    # Execute Units / Value analysis if applicable
    if is_units_format:
        if is_classification_format:
            st.divider()
        render_units_analysis_section(df)


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
