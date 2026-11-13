from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import os
import re
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MOEX_BASE = "https://iss.moex.com/iss"
BOARD = "TQBR"
REQUEST_TIMEOUT = 20
MAX_SEARCH_RESULTS = 5
PRICE_LOOKBACK_DAYS = 14

OFFLINE = os.environ.get("AGENT_OFFLINE", "0") == "1"


def _cache_file(prefix: str, raw_key: str) -> Path:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_key)[:80].strip("_") or "x"
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:10]
    return CACHE_DIR / f"{prefix}_{slug}_{digest}.json"


def _http_get_json(url: str, cache_path: Path) -> dict[str, Any]:
    if OFFLINE:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise RuntimeError(
            f"offline mode: no cached response for this request ({cache_path.name})"
        )

    request = urllib.request.Request(
        url, headers={"User-Agent": "MOEX-Stock-Agent/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8"))
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return payload
    except Exception as exc:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"failed to fetch MOEX data: {exc}") from exc


def _table(payload: dict[str, Any], block_name: str) -> list[dict[str, Any]]:
    block = payload.get(block_name) or {}
    columns = block.get("columns") or []
    rows = block.get("data") or []
    return [dict(zip(columns, row)) for row in rows]


_DATE_FORMATS = ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%d/%m/%Y")


def _parse_date(value: str, field: str = "date") -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"field {field} must contain a date in YYYY-MM-DD format")
    text = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"invalid date '{value}' in field {field}; expected format YYYY-MM-DD"
    )


_TICKER_RE = re.compile(r"^[A-Za-z0-9._-]{1,15}$")


def _resolve_ticker(raw: str) -> str:
    """Normalise a ticker; resolve a company name through the search API."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("ticker is not provided")
    ticker = raw.strip()
    if _TICKER_RE.fullmatch(ticker):
        return ticker.upper()
    found = find_security(ticker)
    if isinstance(found, list) and found:
        return str(found[0]["ticker"])
    raise ValueError(f"ticker '{raw}' not found")


def _candles(ticker: str, day_from: date, day_to: date) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "from": day_from.isoformat(),
            "till": (day_to + timedelta(days=1)).isoformat(),
            "interval": 24,
            "iss.meta": "off",
        }
    )
    url = (
        f"{MOEX_BASE}/engines/stock/markets/shares/boards/{BOARD}"
        f"/securities/{urllib.parse.quote(ticker)}/candles.json?{params}"
    )
    cache_path = _cache_file("candles", f"{ticker}_{day_from}_{day_to}_24")
    payload = _http_get_json(url, cache_path)

    result: list[dict[str, Any]] = []
    for row in _table(payload, "candles"):
        begin = str(row.get("begin") or "")[:10]
        try:
            day = datetime.strptime(begin, "%Y-%m-%d").date()
            close = float(row["close"])
        except (ValueError, TypeError, KeyError):
            continue
        if not (day_from <= day <= day_to):
            continue
        try:
            volume = int(row.get("volume") or 0)
        except (TypeError, ValueError):
            volume = 0
        result.append({"day": day, "close": close, "volume": volume})

    result.sort(key=lambda item: item["day"])
    return result


def find_security(query: str) -> Any:
    try:
        if not isinstance(query, str) or not query.strip():
            return {"error": "query is empty; provide a ticker or a company name"}
        query = query.strip()
        if len(query) > 100:
            return {"error": "search query is too long"}

        params = urllib.parse.urlencode(
            {
                "q": query,
                "engine": "stock",
                "market": "shares",
                "is_trading": 1,
                "iss.meta": "off",
            }
        )
        payload = _http_get_json(
            f"{MOEX_BASE}/securities.json?{params}",
            _cache_file("search", query.lower()),
        )
        rows = _table(payload, "securities")

        # A query like "XXXX" looks like a ticker, not a company name. In that
        # case only an exact secid match is accepted; otherwise the fuzzy
        # search (e.g. "XXXX" -> "XXXXXX") would return a wrong security.
        ticker_like = bool(re.fullmatch(r"[A-Z0-9]{2,5}", query))
        if ticker_like:
            rows = [row for row in rows if str(row.get("secid") or "").upper() == query]

        results: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            ticker = str(row.get("secid") or "").strip()
            if not ticker or ticker in seen:
                continue
            if row.get("is_traded") not in (1, "1", True):
                continue
            seen.add(ticker)
            results.append(
                {
                    "ticker": ticker,
                    "name": str(row.get("shortname") or row.get("name") or "").strip(),
                }
            )
            if len(results) == MAX_SEARCH_RESULTS:
                break
        return results
    except Exception as exc:
        return {"error": f"security search failed: {exc}"}


def get_close_price(ticker: str, date: str) -> Any:
    try:
        resolved = _resolve_ticker(ticker)
        requested = _parse_date(date)
        rows = _candles(
            resolved, requested - timedelta(days=PRICE_LOOKBACK_DAYS), requested
        )
        if not rows:
            return {
                "error": (
                    f"no price data available for {resolved} on "
                    f"{requested.isoformat()}; the date may be in the future, "
                    "a non-trading day, or outside the instrument's history"
                )
            }

        last = rows[-1]
        result: dict[str, Any] = {
            "ticker": resolved,
            "date": last["day"].isoformat(),
            "close": round(last["close"], 2),
        }
        if last["day"] != requested:
            result["requested_date"] = requested.isoformat()
            result["note"] = (
                f"{requested.isoformat()} is a non-trading day; showing the "
                f"close of the nearest preceding trading day "
                f"{last['day'].isoformat()}."
            )
        return result
    except Exception as exc:
        return {"error": f"failed to get the closing price: {exc}"}


def get_history(ticker: str, date_from: str, date_to: str) -> Any:
    try:
        resolved = _resolve_ticker(ticker)
        start = _parse_date(date_from, "date_from")
        end = _parse_date(date_to, "date_to")
        if start > end:
            return {"error": "date_from cannot be later than date_to"}

        rows = _candles(resolved, start, end)
        if not rows:
            return {
                "error": (
                    f"no data available for {resolved} between "
                    f"{start.isoformat()} and {end.isoformat()}"
                )
            }
        return [
            {
                "date": row["day"].isoformat(),
                "close": round(row["close"], 2),
                "volume": row["volume"],
            }
            for row in rows
        ]
    except Exception as exc:
        return {"error": f"failed to get price history: {exc}"}


# Whitelisted AST nodes for the calculator: numbers, + - * /, parentheses.
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """Safely evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("only numbers are allowed")
        return float(value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ValueError("division by zero")
        return _ALLOWED_BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers, the + - * / operators and parentheses are allowed")


def calculate(expression: str) -> Any:
    try:
        if not isinstance(expression, str) or not expression.strip():
            return {"error": "empty expression"}
        if len(expression) > 300:
            return {"error": "expression is too long"}

        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree.body)
        if not math.isfinite(value):
            return {"error": "the result is not a finite number"}
        return round(value, 10)
    except Exception as exc:
        return {"error": f"failed to evaluate the expression: {exc}"}


TOOL_FUNCTIONS = {
    "find_security": find_security,
    "get_close_price": get_close_price,
    "get_history": get_history,
    "calculate": calculate,
}
