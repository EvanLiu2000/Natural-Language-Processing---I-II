from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import tools
from functions import TOOLS
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
TRACES_DIR = BASE_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "deepseek-flash"
MAX_STEPS = 6
REQUEST_TIMEOUT = 60.0

SYSTEM_PROMPT = (
    "You are a stock-market assistant with access to functions for MOEX data. "
    "Rules:\n"
    "1. You do NOT know market prices. Never guess or invent numbers. Every "
    "price, volume, percentage or comparison must come from a tool result.\n"
    "2. If the user gives an exact ticker (for example SBER), use it directly. "
    "If the user names a company instead of a ticker, ALWAYS call find_security "
    "first to confirm the ticker, even if you think you know it.\n"
    "3. Use get_close_price for one date, get_history for a range, and "
    "calculate for ALL arithmetic, including percentages, averages and "
    "comparisons.\n"
    "4. If the request is unclear, ask a clarifying question instead of "
    "calling tools. Example: if no company or ticker is mentioned, ask which "
    "security is meant.\n"
    "5. If the user asks for an action that no tool supports (for example, "
    "buying shares), politely refuse and explain what you can do.\n"
    "6. If a tool returns an error, report it honestly; do not fabricate a "
    "value.\n"
    "7. Answer in the same language as the user's question."
)


class StockAgent:
    def __init__(self) -> None:
        self.model_name = MODEL_NAME
        self.max_steps = MAX_STEPS

        self.client = OpenAI(
            api_key="sk-###############################",
            base_url="https://api.deepseek.com",
            timeout=REQUEST_TIMEOUT,
        )

    def _chat(self, messages: list[dict[str, Any]]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
            stream=False,
        )
        return response.choices[0]

    @staticmethod
    def _assistant_message(message: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        return entry

    @staticmethod
    def _execute_tool(name: str, arguments: dict[str, Any]) -> Any:
        function = tools.TOOL_FUNCTIONS.get(name)
        if function is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return function(**arguments)
        except TypeError as exc:
            return {"error": f"invalid arguments for tool {name}: {exc}"}
        except Exception as exc:
            return {"error": f"tool {name} failed: {exc}"}

    def ask(self, question: str, trace_id: str | None = None) -> dict[str, Any]:
        if trace_id is None:
            trace_id = (
                datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            )
        started_at = datetime.now().isoformat(timespec="seconds")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_log: list[dict[str, Any]] = []
        final_answer: str | None = None
        warning: str | None = None
        api_error: str | None = None
        finish_reason: str | None = None
        step = 0

        for step in range(1, self.max_steps + 1):
            try:
                choice = self._chat(messages)
            except Exception as exc:
                api_error = f"DeepSeek API request failed: {exc}"
                final_answer = f"ERROR: {api_error}"
                warning = final_answer
                break

            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None)
            messages.append(self._assistant_message(message))
            tool_calls = message.tool_calls or []

            if not tool_calls:
                answer = (message.content or "").strip()
                final_answer = answer or "(the model returned an empty answer)"
                break

            # Execute every tool call in this response (parallel tool calls).
            for call in tool_calls:
                name = call.function.name
                raw_arguments = call.function.arguments
                arguments: Any = None
                call_error: str | None = None

                try:
                    arguments = (
                        json.loads(raw_arguments)
                        if isinstance(raw_arguments, str)
                        else raw_arguments
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                except Exception as exc:
                    call_error = f"invalid JSON arguments ({exc}); raw: {raw_arguments}"

                if call_error:
                    result: Any = {"error": f"invalid tool call: {call_error}"}
                else:
                    result = self._execute_tool(name, arguments)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                tool_log.append(
                    {
                        "step": step,
                        "tool_call_id": call.id,
                        "name": name,
                        "arguments": arguments,
                        "raw_arguments": raw_arguments,
                        "result": result,
                    }
                )

        if final_answer is None:
            warning = (
                f"WARNING: the {self.max_steps}-step limit was reached; "
                "the agent could not produce a final answer."
            )
            final_answer = warning

        trace = {
            "id": trace_id,
            "question": question,
            "provider": "deepseek",
            "model": self.model_name,
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "generation": {"temperature": 0, "stream": False},
            "finish_reason": finish_reason,
            "max_steps": self.max_steps,
            "steps_used": step,
            "tool_calls": tool_log,
            "messages": messages,
            "final_answer": final_answer,
            "warning": warning,
            "api_error": api_error,
        }
        trace_path = TRACES_DIR / f"{trace_id}.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {"answer": final_answer, "trace": trace, "trace_path": str(trace_path)}


def main() -> None:
    try:
        agent = StockAgent()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = agent.ask(question)
        print(f"\nAgent: {result['answer']}")
        print(f"[trace: {result['trace_path']}]")


if __name__ == "__main__":
    main()
