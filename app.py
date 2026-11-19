"""
Supply Chain Analytics - Inventory Decision Agent Application
============================================================
Interactive application interface for the Supply Chain Inventory Decision Agent.

Architecture:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Supported Models:
1. Economic Order Quantity (EOQ)
2. Quantity Discount Model
3. Safety Stock & Reorder Point (Continuous Review (s, Q))
4. Fixed-Period Review (Periodic Review / Order-Up-To Level (R, S))
5. Newsvendor Model (Single-Period Probabilistic)

Usage:
  Interactive Menu:   python app.py
  Single Query:       python app.py --question "Our annual demand is 12000 units, ordering cost is $25, holding cost is $2.40. What is our EOQ?"
  Run All 5 Demos:    python app.py --demo
  Run Unit Tests:     python app.py --test
"""

import sys
import argparse
import unittest
from typing import List, Dict, Any

from inventory_agent import InventoryDecisionAgent, AgentDecisionResponse


BENCHMARK_DEMOS = [
    {
        "title": "1. Economic Order Quantity (EOQ)",
        "model": "eoq",
        "question": (
            "Our annual demand is 12,000 units, ordering setup cost is $25 per order, "
            "and unit holding cost is $2.40 per unit per year. Unit cost is $10.00. "
            "Lead time is 5 days. What is our optimal EOQ, annual costs, and reorder point?"
        ),
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
    },
    {
        "title": "3. Safety Stock & Reorder Point (Continuous Review)",
        "model": "safety_stock_rop",
        "question": (
            "Under continuous review (s, Q), our average daily demand is 100 units with a "
            "standard deviation of 10 units. Supplier lead time is 4 days with no lead time variability. "
            "We require a 95% cycle service level. What should our safety stock and reorder point (ROP) be?"
        ),
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
    },
]


def print_banner():
    banner = r"""
================================================================================
     SUPPLY CHAIN ANALYTICS - INVENTORY DECISION AGENT
================================================================================
Architecture:
Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain

Models Supported:
  [1] Economic Order Quantity (EOQ)
  [2] Quantity Discount Model
  [3] Safety Stock & Reorder Point (Continuous Review (s, Q))
  [4] Fixed-Period Review (Periodic Review (R, S))
  [5] Newsvendor Model (Single-Period Probabilistic)
================================================================================
"""
    print(banner)


def run_interactive_mode(agent: InventoryDecisionAgent):
    """Runs the interactive command loop."""
    print_banner()

    while True:
        print("\nMain Menu:")
        print("  [1] Enter a Custom Natural-Language Inventory Question")
        print("  [2] Run Model 1: Economic Order Quantity (EOQ) Demo")
        print("  [3] Run Model 2: Quantity Discount Demo")
        print("  [4] Run Model 3: Safety Stock & Reorder Point Demo")
        print("  [5] Run Model 4: Fixed-Period Review Demo")
        print("  [6] Run Model 5: Newsvendor (Single-Period) Demo")
        print("  [7] Run All 5 Benchmark Demos Sequentially")
        print("  [8] Run Comprehensive Unit Test Suite")
        print("  [Q] Exit Application")

        choice = input("\nSelect an option (1-8 or Q): ").strip().upper()

        if choice in ["Q", "QUIT", "EXIT"]:
            print("\nExiting Supply Chain Inventory Decision Agent. Goodbye!\n")
            break
        elif choice == "1":
            print("\nEnter your inventory question (or type 'back' to return):")
            print("Tip: Mention demand, costs (ordering, holding, unit), lead times, service levels, etc.")
            user_q = input("\nQuestion > ").strip()
            if not user_q or user_q.lower() == "back":
                continue
            try:
                response = agent.process_question(user_q)
                print(response.formatted_output)
            except Exception as e:
                print(f"\n[ERROR] Could not process question: {e}\n")
        elif choice in ["2", "3", "4", "5", "6"]:
            idx = int(choice) - 2
            demo = BENCHMARK_DEMOS[idx]
            print(f"\n>>> Running Scenario: {demo['title']}")
            response = agent.process_question(demo["question"])
            print(response.formatted_output)
        elif choice == "7":
            print("\n>>> Running All 5 Benchmark Supply Chain Scenarios...")
            for idx, demo in enumerate(BENCHMARK_DEMOS, 1):
                print(f"\n[{idx}/5] Scenario: {demo['title']}")
                response = agent.process_question(demo["question"])
                print(response.formatted_output)
        elif choice == "8":
            print("\n>>> Running Unit & Integration Test Suite...")
            suite = unittest.defaultTestLoader.discover("tests")
            runner = unittest.TextTestRunner(verbosity=2)
            runner.run(suite)
        else:
            print(f"\nInvalid option '{choice}'. Please select between 1 and 8 or Q.")


def main():
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Natural-language inventory question to evaluate directly"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run all 5 inventory model showcase demonstrations"
    )
    parser.add_argument(
        "--test", "-t",
        action="store_true",
        help="Execute unit tests for all mathematical tools and agent pipeline"
    )

    args = parser.parse_args()
    agent = InventoryDecisionAgent()

    if args.test:
        suite = unittest.defaultTestLoader.discover("tests")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.demo:
        print_banner()
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

    # Default to interactive session
    run_interactive_mode(agent)


if __name__ == "__main__":
    main()
