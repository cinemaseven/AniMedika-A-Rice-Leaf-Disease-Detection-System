from __future__ import annotations

import re
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-’'][^\W_]+)*", re.UNICODE)

_CATEGORY_LABELS = {
    "en": {
        "monitoring": "Monitoring",
        "sanitation": "Sanitation",
        "treatment": "Treatment",
        "prevention": "Prevention",
        "field_management": "Field Management",
    },
    "fil": {
        "monitoring": "Pagmamanman",
        "sanitation": "Kalinisan sa Bukid",
        "treatment": "Paggamot",
        "prevention": "Pag-iwas",
        "field_management": "Pamamahala sa Bukid",
    },
}

_CATEGORY_RULES = {
    "en": {
        "monitoring": (
            "check", "monitor", "look for", "inspect", "observe", "watch",
            "symptom", "symptoms", "sign", "signs",
        ),
        "sanitation": (
            "remove", "destroy", "bury", "clean", "stubble", "stubbles",
            "infected plant remains", "volunteer rice", "weed", "weeds",
        ),
        "treatment": (
            "fungicide", "pesticide", "insecticide", "spray", "product label",
            "protective clothing", "agricultural technician",
        ),
        "prevention": (
            "avoid", "resistant", "healthy seed", "disease-free", "prevent",
            "control green leafhoppers", "do not plant",
        ),
        "field_management": (
            "fertilizer", "nitrogen", "potassium", "manure", "drain", "water",
            "irrigation", "spacing", "planting", "seedbed", "transplant",
        ),
    },
    "fil": {
        "monitoring": (
            "suriin", "bantayan", "hanapin", "tingnan", "obserbahan",
            "sintomas", "palatandaan",
        ),
        "sanitation": (
            "alisin", "sirain", "ibaon", "linisin", "pinaggapasan",
            "tirang halaman", "volunteer rice", "damo",
        ),
        "treatment": (
            "pamatay-amag", "pesticide", "insektisidyo", "i-spray", "spray",
            "label ng produkto", "proteksiyon", "agricultural technician",
        ),
        "prevention": (
            "iwasan", "matibay", "malusog na binhi", "walang impeksiyon",
            "pigilan", "kontrolin ang green leafhopper", "huwag magtanim",
        ),
        "field_management": (
            "abono", "nitroheno", "potassium", "pataba", "patuyuin", "tubig",
            "irigasyon", "agwat", "pagtatanim", "punlaan", "lipat-tanim",
        ),
    },
}

_STOPWORDS = {
    "en": {
        "a", "an", "and", "are", "as", "at", "be", "because", "by", "for",
        "from", "if", "in", "is", "it", "of", "on", "or", "so", "that",
        "the", "their", "them", "they", "this", "to", "use", "when", "where",
        "with", "will", "your",
    },
    "fil": {
        "ang", "at", "ay", "dahil", "gamit", "gumamit", "habang", "ito",
        "kapag", "kung", "mga", "na", "ng", "para", "sa", "upang",
    },
}

_CATEGORY_PRIORITY = (
    "monitoring",
    "sanitation",
    "treatment",
    "prevention",
    "field_management",
)


class NLPRecommendationError(ValueError):
    """Raised when recommendation text cannot be processed safely."""


def _validate_text(value: str) -> str:
    if not isinstance(value, str):
        raise NLPRecommendationError("Recommendation content must be text.")

    return value


def _tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(value.casefold())


def _normalize_phrase(value: str) -> str:
    return " ".join(_tokenize(value))


def _contains_term(term: str, tokens: list[str], normalized_text: str) -> bool:
    normalized_term = _normalize_phrase(term)
    if not normalized_term:
        return False

    if " " in normalized_term:
        return f" {normalized_term} " in f" {normalized_text} "

    return normalized_term in tokens


def _score_categories(
    tokens: list[str],
    normalized_text: str,
    language: str,
) -> tuple[str, list[str]]:
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}

    for category, terms in _CATEGORY_RULES[language].items():
        matched_terms = [
            term
            for term in terms
            if _contains_term(term, tokens, normalized_text)
        ]
        matches[category] = matched_terms
        scores[category] = sum(3 if " " in term else 1 for term in matched_terms)

    best_category = max(
        _CATEGORY_PRIORITY,
        key=lambda category: (scores[category], -_CATEGORY_PRIORITY.index(category)),
    )

    if scores[best_category] == 0:
        best_category = "field_management"

    return best_category, matches[best_category]


def _extract_keywords(
    tokens: list[str],
    normalized_text: str,
    language: str,
    matched_terms: list[str],
    limit: int = 6,
) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()

    def add_keyword(value: str) -> None:
        normalized = _normalize_phrase(value)
        if normalized and normalized not in seen and len(keywords) < limit:
            seen.add(normalized)
            keywords.append(value)

    for term in matched_terms:
        add_keyword(term)

    all_rule_terms = (
        term
        for category_terms in _CATEGORY_RULES[language].values()
        for term in category_terms
    )
    for term in all_rule_terms:
        if _contains_term(term, tokens, normalized_text):
            add_keyword(term)

    for token in tokens:
        if len(token) > 2 and token not in _STOPWORDS[language]:
            add_keyword(token)

    return keywords


def _format_recommendation(text: str, category_label: str) -> str:
    """Create controlled formatted output without changing the source sentence."""

    return f"{category_label}: {text}"


def _analyze_recommendation(value: str, language: str) -> dict:
    original_text = _validate_text(value)
    tokens = _tokenize(original_text)
    normalized_text = " ".join(tokens)
    category, matched_terms = _score_categories(
        tokens=tokens,
        normalized_text=normalized_text,
        language=language,
    )
    category_label = _CATEGORY_LABELS[language][category]
    keywords = _extract_keywords(
        tokens=tokens,
        normalized_text=normalized_text,
        language=language,
        matched_terms=matched_terms,
    )

    return {
        "text": original_text,
        "formatted_text": _format_recommendation(original_text, category_label),
        "category": category,
        "category_label": category_label,
        "keywords": keywords,
        "matched_terms": matched_terms,
        "token_count": len(tokens),
    }


def build_nlp_recommendations(
    recommendations: Iterable[str],
    season_note: str,
    language: str,
) -> dict:
    """Analyze curated recommendations while preserving their exact wording.

    The deterministic NLP stage tokenizes the existing recommendations, extracts
    keywords, assigns a management category using language-specific rules, and
    creates controlled formatted text. The website continues to display the
    original curated sentences in their original order.
    """

    if language not in {"en", "fil"}:
        raise NLPRecommendationError("Unsupported recommendation language.")

    recommendation_analysis = [
        _analyze_recommendation(item, language)
        for item in recommendations
        if _validate_text(item)
    ]

    return {
        "general_recommendations": [
            analysis["text"] for analysis in recommendation_analysis
        ],
        "recommendation_analysis": recommendation_analysis,
        "season_note": _validate_text(season_note) if season_note else "",
        "nlp_component": {
            "type": "deterministic_rule_based_nlp",
            "language": language,
            "operations": [
                "tokenization",
                "keyword_extraction",
                "rule_based_category_classification",
                "controlled_text_formatting",
            ],
            "content_preserved": True,
        },
    }
