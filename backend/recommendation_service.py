from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache

from backend.config import RECOMMENDATIONS_PATH, RECOMMENDATION_SOURCES_PATH
from backend.nlp_recommendation_service import (
    NLPRecommendationError,
    build_nlp_recommendations,
)

class RecommendationError(RuntimeError):
    """Raised when recommendation content is missing or invalid."""

def determine_season(selected_date: str) -> str:
    try:
        parsed_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise ValueError("selected_date must use the YYYY-MM-DD format.") from exc

    return "wet" if 6 <= parsed_date.month <= 11 else "dry"

@lru_cache(maxsize=1)
def load_recommendations() -> dict:
    if not RECOMMENDATIONS_PATH.exists():
        raise RecommendationError(
            f"Recommendation file is missing: {RECOMMENDATIONS_PATH}"
        )

    with RECOMMENDATIONS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RecommendationError("Recommendation data must be a JSON object.")

    return data

@lru_cache(maxsize=1)
def load_recommendation_sources() -> dict:
    if not RECOMMENDATION_SOURCES_PATH.exists():
        raise RecommendationError(
            f"Recommendation source file is missing: {RECOMMENDATION_SOURCES_PATH}"
        )

    with RECOMMENDATION_SOURCES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RecommendationError("Recommendation source data must be a JSON object.")

    return data

def get_result_bundle(disease_id: str, season: str) -> dict:
    disease = load_recommendations().get(disease_id)
    if not disease:
        raise RecommendationError(
            f"No recommendation content exists for model class '{disease_id}'."
        )
        
    sources = load_recommendation_sources().get(disease_id, [])
    localized = {}
    
    for language in ("en", "fil"):
        content = disease.get(language, {})
        general = content.get("general", [])
        season_note = content.get("season_notes", {}).get(season, "")

        try:
            nlp_result = build_nlp_recommendations(
                recommendations=general,
                season_note=season_note,
                language=language,
            )
        except NLPRecommendationError as exc:
            raise RecommendationError(str(exc)) from exc

        localized[language] = {
            "name": content.get("name", disease_id.replace("_", " ")),
            "description": content.get("description", ""),
            "general_recommendations": nlp_result["general_recommendations"],
            "recommendation_analysis": nlp_result["recommendation_analysis"],
            "season_note": nlp_result["season_note"],
            "sources": sources,
            "nlp_component": nlp_result["nlp_component"],
        }
        
    return localized