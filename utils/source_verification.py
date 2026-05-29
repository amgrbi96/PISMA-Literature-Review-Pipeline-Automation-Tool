"""Source verification via identifier lookup and DOI resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import ResearchConfig
from models.paper import PaperMetadata
from utils.http import RateLimiter, build_session, request_json

logger = logging.getLogger(__name__)


@dataclass
class VerificationVerdict:
    """Verification result for one paper."""

    paper_identity: str = ""
    tier: int = 0
    method: str = ""
    verdict: str = ""
    detail: str = ""


def verify_sources(
        papers: list[PaperMetadata],
        config: ResearchConfig,
) -> tuple[list[PaperMetadata], list[VerificationVerdict]]:
    """Verify papers via Tier 0 (identifier lookup) and Tier 1 (DOI resolution).

    Returns (verified_papers, verdicts). FABRICATED papers are removed from the
    verified list. All verdicts are returned for the audit trail.
    """

    session = build_session("PRISMA-SourceVerification/1.0")
    limiter = RateLimiter(calls_per_second=config.api_settings.semantic_scholar_calls_per_second / 60.0)
    verdicts: list[VerificationVerdict] = []
    fabricated_keys: set[str] = set()

    for paper in papers:
        v = _verify_one(paper, session, limiter, config)
        verdicts.append(v)
        if v.verdict == "FABRICATED":
            fabricated_keys.add(paper.identity_key)
            logger.warning("FABRICATED source detected: '%s' — %s", paper.title, v.detail)

    verified = [p for p in papers if p.identity_key not in fabricated_keys]
    logger.info(
        "Source verification: %s verified, %s fabricated removed, %s total.",
        len(verified),
        len(fabricated_keys),
        len(papers),
    )
    return verified, verdicts


def _verify_one(
        paper: PaperMetadata,
        session: object,
        limiter: RateLimiter,
        config: ResearchConfig,
) -> VerificationVerdict:
    """Run verification tiers for one paper."""

    s2_id = paper.external_ids.get("s2") if paper.external_ids else ""
    pmid = paper.external_ids.get("pubmed") if paper.external_ids else ""

    # Tier 0: Semantic Scholar identifier lookup
    if s2_id:
        v = _check_semantic_scholar(s2_id, paper, session, limiter, config)
        if v:
            return v

    # Tier 0: PubMed ID lookup
    if pmid:
        v = _check_pubmed_id(pmid, paper, session, limiter, config)
        if v:
            return v

    # Tier 1: DOI resolution via Crossref
    if paper.doi:
        v = _check_doi_crossref(paper, session, limiter, config)
        if v:
            return v

    # No identifier available — mark as PLAUSIBLE (no red flags, but can't verify)
    return VerificationVerdict(
        paper_identity=paper.identity_key,
        tier=0,
        method="no_identifier",
        verdict="PLAUSIBLE",
        detail="No identifier available for verification.",
    )


def _check_semantic_scholar(
        s2_id: str,
        paper: PaperMetadata,
        session: object,
        limiter: RateLimiter,
        config: ResearchConfig,
) -> VerificationVerdict | None:
    """Tier 0: Look up paper on Semantic Scholar by S2 ID."""

    try:
        payload = request_json(
            session,
            "GET",
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_id}",
            limiter=limiter,
            timeout=config.request_timeout_seconds,
            params={"fields": "title,externalIds"},
        )
        if payload and payload.get("title"):
            return VerificationVerdict(
                paper_identity=paper.identity_key,
                tier=0,
                method="semantic_scholar",
                verdict="VERIFIED",
                detail=f"S2 ID {s2_id} confirmed: '{payload['title']}'",
            )
        if payload is not None:
            return VerificationVerdict(
                paper_identity=paper.identity_key,
                tier=0,
                method="semantic_scholar",
                verdict="UNVERIFIABLE",
                detail=f"S2 ID {s2_id} returned empty result.",
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("S2 lookup failed for %s: %s", s2_id, exc)
    return None


def _check_pubmed_id(
        pmid: str,
        paper: PaperMetadata,
        session: object,
        limiter: RateLimiter,
        config: ResearchConfig,
) -> VerificationVerdict | None:
    """Tier 0: Look up paper on PubMed by PMID."""

    try:
        payload = request_json(
            session,
            "GET",
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            limiter=limiter,
            timeout=config.request_timeout_seconds,
            params={"db": "pubmed", "retmode": "json", "term": pmid},
        )
        if payload:
            ids = payload.get("esearchresult", {}).get("idlist", [])
            if pmid in ids:
                return VerificationVerdict(
                    paper_identity=paper.identity_key,
                    tier=0,
                    method="pubmed",
                    verdict="VERIFIED",
                    detail=f"PMID {pmid} confirmed in PubMed.",
                )
            return VerificationVerdict(
                paper_identity=paper.identity_key,
                tier=0,
                method="pubmed",
                verdict="UNVERIFIABLE",
                detail=f"PMID {pmid} not found in PubMed.",
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("PubMed lookup failed for PMID %s: %s", pmid, exc)
    return None


def _check_doi_crossref(
        paper: PaperMetadata,
        session: object,
        limiter: RateLimiter,
        config: ResearchConfig,
) -> VerificationVerdict | None:
    """Tier 1: Resolve DOI via Crossref."""

    try:
        payload = request_json(
            session,
            "GET",
            f"https://api.crossref.org/works/{paper.doi}",
            limiter=limiter,
            timeout=config.request_timeout_seconds,
        )
        if payload and payload.get("status") == "ok":
            msg = payload.get("message", {})
            cr_title = ""
            if msg.get("title"):
                cr_title = msg["title"][0] if isinstance(msg["title"], list) else str(msg["title"])
            return VerificationVerdict(
                paper_identity=paper.identity_key,
                tier=1,
                method="crossref_doi",
                verdict="VERIFIED",
                detail=f"DOI {paper.doi} resolved via Crossref: '{cr_title}'",
            )
    except Exception as exc:  # noqa: BLE001
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code == 404:
            return VerificationVerdict(
                paper_identity=paper.identity_key,
                tier=1,
                method="crossref_doi",
                verdict="UNVERIFIABLE",
                detail=f"DOI {paper.doi} returned 404 from Crossref.",
            )
        logger.debug("Crossref DOI lookup failed for %s: %s", paper.doi, exc)
    return None
