TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_security",
            "description": (
                "Searches for stocks on the Moscow Exchange (MOEX) by ticker "
                "or company name. The query argument is a string with a "
                'ticker or a name, for example "SBER" or "Sberbank". '
                'Example call: {"query": "Sberbank"}. '
                "Returns at most 5 objects of the form {ticker, name}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ('Ticker or company name. Example: "Sberbank".'),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_close_price",
            "description": (
                "Returns the closing price of a stock for the specified date. "
                'Arguments: ticker — stock ticker (for example, "SBER"); '
                "date — date in YYYY-MM-DD format (for example, "
                '"2026-09-15"). If the date is not a trading day, the tool '
                "returns the nearest preceding trading day and a note field "
                "with an explanation. "
                'Example call: {"ticker": "SBER", "date": '
                '"2026-09-15"}.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": 'Stock ticker. Example: "SBER".',
                    },
                    "date": {
                        "type": "string",
                        "description": (
                            'Date in YYYY-MM-DD format. Example: "2026-09-15".'
                        ),
                    },
                },
                "required": ["ticker", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Returns daily closing prices and trading volumes for a "
                "period. Arguments: ticker — stock ticker; date_from and "
                "date_to — dates in YYYY-MM-DD format, with date_from not "
                "later than date_to. "
                'Example call: {"ticker": "SBER", "date_from": '
                '"2026-09-01", "date_to": "2026-09-30"}. '
                "Returns a list of objects {date, close, volume}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": 'Stock ticker. Example: "SBER".',
                    },
                    "date_from": {
                        "type": "string",
                        "description": (
                            "Start of the period in YYYY-MM-DD format. "
                            'Example: "2026-09-01".'
                        ),
                    },
                    "date_to": {
                        "type": "string",
                        "description": (
                            "End of the period in YYYY-MM-DD format. "
                            'Example: "2026-09-30".'
                        ),
                    },
                },
                "required": ["ticker", "date_from", "date_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Performs an arithmetic calculation. Only numbers, the "
                "operators + - * / and parentheses are allowed. The expression "
                "argument is a string containing the expression. "
                'Example call: {"expression": '
                '"(284.91 - 272.21) / 272.21 * 100"}.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Arithmetic expression, for example "
                            '"(285 - 272) / 272 * 100".'
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
]
