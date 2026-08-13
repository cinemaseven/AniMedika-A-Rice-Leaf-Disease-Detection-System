from __future__ import annotations

import re
from collections.abc import Iterable

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)

# Language-level symptom/sign patterns. Matches are extracted directly from the
# verified description text; the component does not generate new disease facts.
_KEYPHRASE_PATTERNS = {
    "en": (
        r"\byellow\s+to\s+white\s+(?:lesions?|spots?)\b",
        r"\bwater-soaked\s+(?:streaks?|lesions?|spots?)\b",
        r"\bmilky\s+or\s+cloudy\s+bacterial\s+droplets\b",
        r"\bbrown\s+(?:spots|margins?)\b",
        r"\boval-shaped(?:\s+spots?)?\b",
        r"\bgrayish\s+or\s+whitish\s+center\b",
        r"\byellowish\s+halo\b",
        r"\bspindle-shaped\s+lesions\b",
        r"\bgray\s+centers\b",
        r"\bgreenish-gray\b",
        r"\bwater-soaked\b",
        r"\bstunted\b",
        r"\bfewer\s+tillers\b",
        r"\bshortened\s+leaf\s+sheaths\s+and\s+leaf\s+blades\b",
        r"\byellow\s+to\s+orange-yellow\s+discoloration(?:\s+of\s+the\s+leaves)?\b",
        r"\bmottling\b",
        r"\byellow\s+streaks\b",
        r"\bfresh-looking\b",
        r"\bfree\s+from\s+visible\s+signs\s+of\s+disease\s+or\s+damage\b",
        r"\bgreen\b",
    ),
    "fil": (
        r"\bdilaw\s+hanggang\s+puting\s+sugat\s+o\s+lesyon\b",
        r"\btila\s+basang\s+guhit\b",
        r"\bparang\s+gatas\s+o\s+malabong\s+patak\s+ng\s+bakterya\b",
        r"\bkayumangging\s+batik\b",
        r"\bhugis-itlog(?:\s+na\s+batik)?\b",
        r"\bgitnang\s+abuhin\s+o\s+maputi\b",
        r"\bmadilaw\s+na\s+halo\b",
        r"\bpahaba\s+at\s+patulis\s+na\s+batik\b",
        r"\babuhing\s+gitna\b",
        r"\bkayumangging\s+gilid\b",
        r"\bmamerdeng-abuhing\b",
        r"\bbasang\b",
        r"\bnababansot\b",
        r"\bkakaunti\s+ang\s+suwi\b",
        r"\bumiikli\s+ang\s+lapak\s+at\s+balat-dahon\b",
        r"\bnaninilaw\s+hanggang\s+kahel-dilaw\s+ang\s+mga\s+dahon\b",
        r"\bbatik-batik\b",
        r"\bguhit\s+na\s+maputla\b",
        r"\bsariwa\s+ang\s+itsura\b",
        r"\bwalang\s+nakikitang\s+palatandaan\s+ng\s+sakit\s+o\s+pinsala\b",
        r"\bberde\b",
    ),
}


class NLPKeyphraseError(ValueError):
    """Raised when disease description text cannot be processed safely."""


def _validate_description(description: str | Iterable[str]) -> list[str]:
    if isinstance(description, str):
        return [description]

    if not isinstance(description, Iterable):
        raise NLPKeyphraseError("Disease description must be text or a list of text.")

    paragraphs = list(description)

    if any(not isinstance(item, str) for item in paragraphs):
        raise NLPKeyphraseError("Every disease description paragraph must be text.")

    return paragraphs


def _tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(value.casefold())


def _normalize_phrase(value: str) -> str:
    return " ".join(_tokenize(value))


def extract_key_phrases(
    description: str | Iterable[str],
    language: str,
    limit: int = 3,
) -> dict:
    """Extract key signs directly from the existing verified description text.

    This lightweight deterministic NLP stage normalizes and tokenizes the disease
    description, then applies language-specific keyphrase patterns. Every returned
    phrase is an exact substring of the original description, so no new disease
    information is generated or rewritten.
    """

    if language not in _KEYPHRASE_PATTERNS:
        raise NLPKeyphraseError("Unsupported keyphrase language.")

    paragraphs = _validate_description(description)
    text = " ".join(paragraph.strip() for paragraph in paragraphs if paragraph.strip())

    if not text:
        return {
            "key_phrases": [],
            "nlp_component": {
                "type": "deterministic_rule_based_keyphrase_extraction",
                "language": language,
                "operations": [
                    "text_normalization",
                    "tokenization",
                    "rule_based_keyphrase_extraction",
                ],
                "content_preserved": True,
                "token_count": 0,
            },
        }

    matches: list[tuple[int, int, str]] = []

    for pattern in _KEYPHRASE_PATTERNS[language]:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.UNICODE):
            matches.append((match.start(), match.end(), match.group(0)))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    key_phrases: list[str] = []
    seen: set[str] = set()

    for _start, _end, phrase in matches:
        normalized = _normalize_phrase(phrase)

        if not normalized or normalized in seen:
            continue

        if any(
            normalized in existing or existing in normalized
            for existing in seen
        ):
            continue

        seen.add(normalized)
        key_phrases.append(phrase)

        if len(key_phrases) >= limit:
            break

    return {
        "key_phrases": key_phrases,
        "nlp_component": {
            "type": "deterministic_rule_based_keyphrase_extraction",
            "language": language,
            "operations": [
                "text_normalization",
                "tokenization",
                "rule_based_keyphrase_extraction",
            ],
            "content_preserved": True,
            "token_count": len(_tokenize(text)),
        },
    }
