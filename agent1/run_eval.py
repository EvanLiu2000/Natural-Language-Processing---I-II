from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tools
from agent import MODEL_NAME, TRACES_DIR, StockAgent, _configure_stdout

BASE_DIR = Path(__file__).resolve().parent

# The 12 mandatory questions from the assignment.
TEST_CASES = [
    {
        "id": "q01",
        "question": "How much did Sberbank's share cost at the close on September 15, 2026?",
        "expected_tools": ["find_security", "get_close_price"],
        "expected_behavior": "find_security -> get_close_price",
    },
    {
        "id": "q02",
        "question": "Closing price of SBER and GAZP on 09/15/2026",
        "expected_tools": ["get_close_price", "get_close_price"],
        "expected_behavior": "two separate calls to get_close_price",
    },
    {
        "id": "q03",
        "question": "How many percent did SBER grow from 09/01/2026 to 30.09.2026?",
        "expected_tools": ["get_history", "calculate"],
        "expected_behavior": "get_history -> calculate",
    },
    {
        "id": "q04",
        "question": "Which day in September 2026 was the most expensive for LKOH?",
        "expected_tools": ["get_history"],
        "expected_behavior": "get_history, find maximum price from returned dataset",
    },
    {
        "id": "q05",
        "question": "Average SBER trading volume for September 2026",
        "expected_tools": ["get_history", "calculate"],
        "expected_behavior": "get_history -> calculate",
    },
    {
        "id": "q06",
        "question": "Which is more expensive: 10 shares of SBER or 3 shares of ROSN on 09/15/2026?",
        "expected_tools": ["get_close_price", "get_close_price", "calculate"],
        "expected_behavior": "2 x get_close_price -> calculate",
    },
    {
        "id": "q07",
        "question": "Share price XXXX 09/15/2026",
        "expected_tools": [],
        "expected_behavior": "tool error -> honest answer 'ticker not found', no invented price",
    },
    {
        "id": "q08",
        "question": "SBER price on September 13, 2026 (Saturday)",
        "expected_tools": ["get_close_price"],
        "expected_behavior": "nearest trading day data plus explanatory note",
    },
    {
        "id": "q09",
        "question": "SBER price as of January 1, 2030",
        "expected_tools": ["get_close_price"],
        "expected_behavior": "disclaimer: no available data for this date",
    },
    {
        "id": "q10",
        "question": "How much does the promotion cost?",
        "expected_tools": [],
        "expected_behavior": "ask clarifying question, do not invoke tools",
    },
    {
        "id": "q11",
        "question": "Hi, what can you do?",
        "expected_tools": [],
        "expected_behavior": "direct chat-style reply, do not invoke tools",
    },
    {
        "id": "q12",
        "question": "Buy me 10 shares of SBER",
        "expected_tools": [],
        "expected_behavior": "refusal: no corresponding tool",
    },
]


def _extract_numbers(text: str) -> list[float]:
    """Extract numbers from an answer, handling comma decimals and thousands."""
    numbers: list[float] = []
    variants = [text, text.replace(",", "."), text.replace(",", "")]
    for variant in variants:
        for match in re.finditer(r"-?\d+(?:\.\d+)?", variant):
            try:
                numbers.append(float(match.group(0)))
            except ValueError:
                continue
    return numbers


def _mentions_value(answer: str, value: Any, rel_tol: float = 5e-3) -> bool:
    if not isinstance(value, (int, float)):
        return False
    candidates = _extract_numbers(answer)
    for candidate in candidates:
        if abs(candidate - value) <= max(abs(value) * rel_tol, 1e-9):
            return True

    lowered = answer.lower()
    scales = (
        (1e6, ("million", "mln")),
        (1e3, ("thousand",)),
    )
    for scale, words in scales:
        if any(word in lowered for word in words):
            for candidate in candidates:
                if abs(candidate * scale - value) <= max(abs(value) * rel_tol, 1e-9):
                    return True
    return any(
        formatted in answer.replace(",", ".")
        for formatted in (f"{float(value):.2f}", f"{float(value):.4f}")
    )


def _mentions_date(answer: str, iso_date: str) -> bool:
    """Check several common date spellings."""
    try:
        day = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError:
        return iso_date in answer
    variants = [
        iso_date,
        day.strftime("%d.%m.%Y"),
        day.strftime("%m/%d/%Y"),
        day.strftime("%B %d, %Y").replace(" 0", " "),
        day.strftime("%d %B %Y").replace(" 0", " "),
        day.strftime("%b %d, %Y").replace(" 0", " "),
    ]
    lowered = answer.lower()
    return any(variant.lower() in lowered for variant in variants)


def _tool_names(trace: dict[str, Any]) -> list[str]:
    return [call["name"] for call in trace.get("tool_calls", []) if call.get("name")]


def _results_by_name(trace: dict[str, Any], name: str) -> list[Any]:
    return [
        call["result"]
        for call in trace.get("tool_calls", [])
        if call.get("name") == name
    ]


def _successful_prices(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in _results_by_name(trace, "get_close_price")
        if isinstance(result, dict) and "close" in result
    ]


def _numeric_results(trace: dict[str, Any], name: str) -> list[float]:
    return [
        float(result)
        for result in _results_by_name(trace, name)
        if isinstance(result, (int, float)) and not isinstance(result, bool)
    ]


def assess(case: dict[str, Any], trace: dict[str, Any]) -> tuple[str, str]:
    """Return ("yes" | "no" | "review", short reason)."""
    qid = case["id"]
    answer = trace.get("final_answer") or ""
    lowered = answer.lower()
    names = _tool_names(trace)
    expected = case["expected_tools"]

    # Questions 10-12 must not call any tool.
    if qid in {"q10", "q11", "q12"}:
        if names:
            return "no", f"unexpected tool calls: {names}"
        if qid == "q10":
            asks = "?" in answer and any(
                word in lowered
                for word in (
                    "which",
                    "what",
                    "company",
                    "ticker",
                    "security",
                    "stock",
                    "share",
                )
            )
            return (
                ("yes", "clarifying question")
                if asks
                else ("review", "no clear clarification")
            )
        if qid == "q11":
            describes = any(
                word in lowered
                for word in (
                    "price",
                    "stock",
                    "history",
                    "help",
                    "find",
                    "calculate",
                    "volume",
                    "ticker",
                )
            )
            return (
                ("yes", "capability description")
                if describes
                else ("review", "unclear description")
            )
        refuses = any(
            phrase in lowered
            for phrase in (
                "cannot",
                "can't",
                "can not",
                "unable",
                "not able",
                "no tool",
                "don't have",
                "do not have",
            )
        )
        return ("yes", "refusal") if refuses else ("review", "no explicit refusal")

    # Question 7: must not produce a price for the unknown ticker.
    if qid == "q07":
        prices = _successful_prices(trace)
        not_found = any(
            phrase in lowered
            for phrase in (
                "not found",
                "not be found",
                "no such",
                "unknown",
                "couldn't find",
                "could not find",
                "not available",
                "does not exist",
                "no match",
                "no results",
                "placeholder",
                "couldn't identify",
                "could not identify",
                "can't identify",
                "cannot identify",
                "not a real ticker",
                "invalid ticker",
            )
        )
        if prices:
            return "no", "a price was returned for a non-existent ticker"
        return (
            ("yes", "honest not-found answer")
            if not_found
            else ("review", "no explicit not-found wording")
        )

    # Question 9: no historical data for 2030.
    if qid == "q09":
        prices = _successful_prices(trace)
        no_data = any(
            phrase in lowered
            for phrase in (
                "no data",
                "no available",
                "not available",
                "unavailable",
                "future",
                "cannot",
                "can't",
            )
        )
        if prices:
            return "no", "a price was returned for a future date"
        return (
            ("yes", "no-data disclaimer")
            if no_data
            else ("review", "no explicit disclaimer")
        )

    # All other questions: verify the expected tool calls.
    missing = [
        tool for tool in set(expected) if names.count(tool) < expected.count(tool)
    ]
    if missing:
        return "no", f"missing expected tool calls: {missing}"

    if qid == "q01":
        prices = _successful_prices(trace)
        if prices and _mentions_value(answer, prices[-1]["close"]):
            return "yes", "tool result quoted in the answer"
        return "review", "tool called but the price was not found in the answer"

    if qid == "q02":
        prices = _successful_prices(trace)
        if len(prices) >= 2 and all(
            _mentions_value(answer, p["close"]) for p in prices[:2]
        ):
            return "yes", "both closing prices quoted"
        return "review", "two prices requested, but both were not found in the answer"

    if qid == "q03":
        calc = _numeric_results(trace, "calculate")
        if calc and _mentions_value(answer, calc[-1]):
            return "yes", "percentage computed with calculate"
        return "review", "calculate called but result not found in the answer"

    if qid == "q04":
        history = [
            result
            for result in _results_by_name(trace, "get_history")
            if isinstance(result, list) and result
        ]
        if not history:
            return "review", "no history data"
        rows = [row for batch in history for row in batch if isinstance(row, dict)]
        if not rows:
            return "review", "empty history data"
        best = max(rows, key=lambda row: row.get("close", float("-inf")))
        if _mentions_date(answer, best["date"]) or _mentions_value(
            answer, best["close"]
        ):
            return "yes", f"max close on {best['date']}"
        return "review", f"expected the maximum on {best['date']}"

    if qid == "q05":
        calc = _numeric_results(trace, "calculate")
        if calc and _mentions_value(answer, calc[-1]):
            return "yes", "average computed with calculate"
        return "review", "calculate called but result not found in the answer"

    if qid == "q06":
        calc = _numeric_results(trace, "calculate")
        if calc and _mentions_value(answer, calc[-1]):
            return "yes", "comparison computed with calculate"
        return "review", "calculate called but result not found in the answer"

    if qid == "q08":
        prices = _successful_prices(trace)
        if not prices:
            return "no", "no price returned"
        price = prices[-1]
        has_price = _mentions_value(answer, price["close"])
        has_note = "note" in price and (
            _mentions_date(answer, price["date"])
            or any(
                word in lowered
                for word in (
                    "weekend",
                    "non-trading",
                    "not a trading",
                )
            )
        )
        if has_price and has_note:
            return "yes", "nearest trading day and note reported"
        return "review", "price or explanatory note not found in the answer"

    return "review", "manual check required"


def _short(value: Any, limit: int = 240) -> str:
    text = (
        json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    )
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("|", "\\|")


def _calls_summary(trace: dict[str, Any]) -> str:
    calls = trace.get("tool_calls", [])
    if not calls:
        return "—"
    parts = [
        f"{call.get('name')}({_short(call.get('arguments'), 120)})" for call in calls
    ]
    return "<br>".join(parts)


def _write_markdown_table(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "| No. | Called Tools | Arguments | Agent's response | Correct (yes/no) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in results:
        lines.append(
            "| {id} | {tools} | {args} | {answer} | {verdict} |".format(
                id=item["id"],
                tools=item["tools"] or "—",
                args=item["arguments"] or "—",
                answer=_short(item["answer"], 300),
                verdict=item["verdict"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _configure_stdout()
    print(f"Initializing DeepSeek agent (model: {MODEL_NAME})")
    try:
        agent = StockAgent()
    except Exception as exc:
        print(f"ERROR: {exc}")
        return

    results: list[dict[str, Any]] = []
    for case in TEST_CASES:
        print(f"\n[{case['id']}] {case['question']}")
        run = agent.ask(case["question"], trace_id=case["id"])
        trace = run["trace"]
        verdict, reason = assess(case, trace)
        names = _tool_names(trace)

        arguments = (
            "<br>".join(
                _short(call.get("arguments"), 160)
                for call in trace.get("tool_calls", [])
            )
            or "—"
        )

        item = {
            "id": case["id"],
            "question": case["question"],
            "expected_behavior": case["expected_behavior"],
            "tools": ", ".join(names) if names else "—",
            "arguments": arguments,
            "answer": run["answer"],
            "verdict": verdict,
            "reason": reason,
            "trace_path": run["trace_path"],
        }
        results.append(item)

        print(f"  tools: {names or '—'}")
        print(f"  answer: {_short(run['answer'], 500)}")
        print(f"  verdict: {verdict} ({reason})")

    # Save machine-readable and Markdown summaries.
    summary_path = TRACES_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": agent.model_name,
                "max_steps": agent.max_steps,
                "offline": tools.OFFLINE,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    table_path = TRACES_DIR / "results_table.md"
    _write_markdown_table(results, table_path)

    yes = sum(1 for item in results if item["verdict"] == "yes")
    review = sum(1 for item in results if item["verdict"] == "review")
    no = sum(1 for item in results if item["verdict"] == "no")
    print(f"\nSummary: {yes} yes, {no} no, {review} review")
    print(f"Table: {table_path}")
    print(f"Details: {summary_path}")


if __name__ == "__main__":
    main()
