from fractions import Fraction
import re

UNIT_NUMBER_RE = re.compile(
    r"(?<!\w)(\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
    r"(?=\s*(?:cups?|tbsps?|tablespoons?|tsps?|teaspoons?|oz|ounces?|lbs?|pounds?|"
    r"grams?|g|kg|ml|l|cloves?|pinch(?:es)?|pitted|large|medium|small|cans?)\b)",
    re.IGNORECASE,
)

COMMON_DENOMINATORS = (2, 3, 4, 8)


def _parse_qty(token: str) -> float:
    token = token.strip()
    if " " in token and "/" in token:
        whole, frac = token.split(None, 1)
        return float(whole) + float(Fraction(frac))
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def format_kitchen_qty(value: float) -> str:
    if value <= 0:
        return "0"
    whole = int(value)
    rem = value - whole
    if rem < 0.04:
        return str(whole) if whole > 0 else "pinch"

    best_frac = None
    best_err = 0.04
    for den in COMMON_DENOMINATORS:
        num = round(rem * den)
        if num == 0:
            continue
        if num == den:
            return str(whole + 1)
        err = abs(rem - (num / den))
        if err < best_err:
            best_err = err
            best_frac = Fraction(num, den)

    if best_frac is not None:
        return f"{whole} {best_frac}" if whole > 0 else str(best_frac)
    return f"{value:.2f}".rstrip("0").rstrip(".")


def scale_substitution_text(text: str, multiplier: float = 1.0) -> str:
    if abs(multiplier - 1.0) < 1e-6:
        return text

    def _repl(match: re.Match) -> str:
        raw = match.group(1)
        scaled = _parse_qty(raw) * multiplier
        return format_kitchen_qty(scaled)

    return UNIT_NUMBER_RE.sub(_repl, text)
