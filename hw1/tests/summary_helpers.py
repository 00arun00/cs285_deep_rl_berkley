"""Read semantic values from rendered, unwrapped summary rows."""

import re


def summary_row_value(rendered: str, label: str) -> str:
    for line in rendered.splitlines():
        # Discard table borders and normalize spacing, without fixing row order.
        text = " ".join(re.sub(r"[\u2500-\u257f|]", " ", line).split())
        if text.startswith(label + " "):
            return text[len(label) :].strip()
    raise AssertionError(f"Summary row {label!r} was not rendered")


def summary_row_numbers(rendered: str, label: str) -> list[int]:
    value = summary_row_value(rendered, label)
    return [int(number.replace(",", "")) for number in re.findall(r"\d[\d,]*", value)]
