"""Search reproducibility verification by re-executing searches and comparing counts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from config import ResearchConfig
from models.paper import PaperMetadata
from utils.http import RateLimiter, build_session, request_json

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of re-executing one source's search for comparison."""

    source: str = ""
    original_count: int = 0
    verification_count: int = 0
    discrepancy_pct: float = 0.0
    classification: str = ""
    detail: str = ""


def verify_search_reproducibility(
        source_records: list[dict[str, Any]],
        config: ResearchConfig,
) -> list[VerificationResult]:
    """Re-execute PubMed searches and compare counts against original results.

    Only PubMed is verified (the primary medical database with a stable API).
    Other sources are documented as unverified.
    """

    results: list[VerificationResult] = []
    session = build_session("PRISMA-SourceVerification/1.0")
    limiter = RateLimiter(calls_per_second=config.api_settings.pubmed_calls_per_second)

    for record in source_records:
        source = record.get("source", "")
        original_count = record.get("results_returned", 0)

        if source == "pubmed":
            v = _verify_pubmed(record, original_count, session, limiter, config)
            results.append(v)
        else:
            results.append(VerificationResult(
                source=source,
                original_count=original_count,
                verification_count=original_count,
                discrepancy_pct=0.0,
                classification="NOT_VERIFIED",
                detail=f"Reproducibility verification not implemented for {source}.",
            ))

    verified_count = sum(1 for r in results if r.classification == "SEARCH_VERIFIED")
    logger.info(
        "Search reproducibility: %s/%s sources verified, %s approximate, %s unverified.",
        verified_count,
        len(results),
        sum(1 for r in results if r.classification == "SEARCH_APPROXIMATE"),
        sum(1 for r in results if r.classification in {"SEARCH_UNVERIFIED", "NOT_VERIFIED"}),
    )
    return results


def _verify_pubmed(
        record: dict[str, Any],
        original_count: int,
        session: object,
        limiter: RateLimiter,
        config: ResearchConfig,
) -> VerificationResult:
    """Re-execute PubMed search and compare result count."""

    query_variants = record.get("query_variants", [])
    if not query_variants:
        return VerificationResult(
            source="pubmed",
            original_count=original_count,
            classification="NOT_VERIFIED",
            detail="No query variants recorded for PubMed.",
        )

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    total_verified = 0

    for query in query_variants:
        search_term = (
            f"({query}) AND "
            f"({config.year_range_start}:{config.year_range_end}[pdat])"
        )
        try:
            payload = request_json(
                session,
                "GET",
                search_url,
                limiter=limiter,
                timeout=config.request_timeout_seconds,
                params={
                    "db": "pubmed",
                    "retmode": "json",
                    "retmax": 0,
                    "term": search_term,
                },
            )
            if payload:
                count_str = payload.get("esearchresult", {}).get("count", "0")
                total_verified += int(count_str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PubMed reproducibility check failed for query '%s': %s", query[:80], exc)
            return VerificationResult(
                source="pubmed",
                original_count=original_count,
                classification="NOT_VERIFIED",
                detail=f"Re-execution failed: {exc}",
            )

    if original_count == 0 and total_verified == 0:
        discrepancy = 0.0
    elif original_count == 0:
        discrepancy = 100.0
    else:
        discrepancy = abs(total_verified - original_count) / max(original_count, 1) * 100.0

    if discrepancy <= 5.0:
        classification = "SEARCH_VERIFIED"
    elif discrepancy <= 20.0:
        classification = "SEARCH_APPROXIMATE"
    else:
        classification = "SEARCH_UNVERIFIED"

    return VerificationResult(
        source="pubmed",
        original_count=original_count,
        verification_count=total_verified,
        discrepancy_pct=round(discrepancy, 1),
        classification=classification,
        detail=f"Original: {original_count}, Re-executed: {total_verified}, Discrepancy: {discrepancy:.1f}%",
    )
