import os
from typing import Dict, List, Tuple, Optional


def preprocess_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _format_plan_lines(orders: Dict[str, int]) -> List[str]:
    lines = []
    for product, qty in orders.items():
        if qty <= 0:
            lines.append(f"- {product}: no order needed")
        else:
            lines.append(f"- {product}: order {qty} units")
    return lines


def _template_report(plan: Dict[str, object]) -> str:
    orders = plan.get("orders", {})
    total_cost = plan.get("total_cost", "N/A")
    budget_remaining = plan.get("budget_remaining", "N/A")
    service_level = plan.get("service_level", "N/A")
    notes = plan.get("notes", "")

    lines = [
        "Inventory Replenishment Summary",
        "",
        "Recommended orders:",
        *_format_plan_lines(orders),
        "",
        f"Total cost (estimated): {total_cost}",
        f"Budget remaining: {budget_remaining}",
        f"Service level target: {service_level}",
    ]
    if notes:
        lines.extend(["", f"Notes: {notes}"])

    return "\n".join(lines)


def generate_report(plan: Dict[str, object]) -> str:
    # Optional LLM mode: enable only if configured and dependency exists.
    if os.getenv("LLM_MODE", "").lower() == "openai":
        try:
            from openai import OpenAI  # type: ignore

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            prompt = (
                "Summarize the inventory plan in clear business language. "
                "Include orders, total cost, budget remaining, and service level.\n\n"
                f"Plan: {plan}"
            )
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt,
            )
            return response.output_text.strip()
        except Exception:
            # Fall back to template if LLM is not available or fails.
            pass

    return _template_report(plan)


def evaluate_report(report: str, plan: Dict[str, object]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    normalized = preprocess_text(report)
    orders = plan.get("orders", {})

    for product in orders.keys():
        if preprocess_text(product) not in normalized:
            issues.append(f"Missing product mention: {product}")

    if "total" not in normalized:
        issues.append("Missing total cost mention")

    return (len(issues) == 0), issues
