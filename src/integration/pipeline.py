from typing import Dict

from src.search_csp.agent import InventoryAgent
from src.search_csp.algorithm import backtracking
from src.nlp.nlp_component import generate_report, evaluate_report


def _simulate_predictions() -> Dict[str, int]:
    return {"A": 80, "B": 40, "C": 90}


def _build_initial_state(predicted_demand: Dict[str, int]) -> Dict[str, object]:
    return {
        "inventario": {"A": 50, "B": 30, "C": 100},
        "demanda": predicted_demand,
        "presupuesto": 10000,
    }


def _build_plan(initial_state: Dict[str, object], final_state: Dict[str, object], costos: Dict[str, int]) -> Dict[str, object]:
    orders = {}
    total_cost = 0
    for product, initial_qty in initial_state["inventario"].items():
        final_qty = final_state["inventario"][product]
        order_qty = max(0, final_qty - initial_qty)
        orders[product] = order_qty
        total_cost += order_qty * costos[product]

    budget_remaining = final_state["presupuesto"]
    plan = {
        "orders": orders,
        "total_cost": total_cost,
        "budget_remaining": budget_remaining,
        "service_level": "95%",
        "notes": "Plan generated from CSP backtracking with simulated demand.",
    }
    return plan


def run_pipeline(simulated: bool = True) -> Dict[str, object]:
    productos = ["A", "B", "C"]
    costos = {"A": 50, "B": 100, "C": 30}
    capacidad = 500

    predicted_demand = _simulate_predictions() if simulated else _simulate_predictions()
    initial_state = _build_initial_state(predicted_demand)

    agent = InventoryAgent(productos, costos, capacidad)
    final_state = backtracking(initial_state, productos, costos, capacidad)

    if not final_state:
        raise RuntimeError("No solution found for CSP inventory plan")

    plan = _build_plan(initial_state, final_state, costos)
    report = generate_report(plan)
    ok, issues = evaluate_report(report, plan)

    result = {
        "plan": plan,
        "report": report,
        "report_ok": ok,
        "report_issues": issues,
        "is_goal_state": agent.es_meta(final_state),
        "final_state": final_state,
    }
    return result


if __name__ == "__main__":
    output = run_pipeline(simulated=True)
    print("=== NLP REPORT ===")
    print(output["report"])
    if not output["report_ok"]:
        print("\nReport issues:")
        for issue in output["report_issues"]:
            print(f"- {issue}")
