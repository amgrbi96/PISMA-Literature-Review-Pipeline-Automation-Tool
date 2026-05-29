"""Deduplication helpers for merging overlapping records across discovery sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models.paper import PaperMetadata


@dataclass
class DeduplicationAuditEntry:
    """One recorded duplicate pair from the deduplication process."""

    kept_id: str = ""
    removed_id: str = ""
    kept_source: str = ""
    removed_source: str = ""
    method: str = ""
    similarity: float = 0.0
    reason: str = ""


@dataclass
class DeduplicationResult:
    """Structured output from deduplication with full audit trail."""

    unique: list[PaperMetadata] = field(default_factory=list)
    duplicates: list[PaperMetadata] = field(default_factory=list)
    audit_trail: list[DeduplicationAuditEntry] = field(default_factory=list)


def deduplicate_papers(
        papers: Iterable[PaperMetadata],
        *,
        title_similarity_threshold: float = 0.92,
) -> list[PaperMetadata]:
    """Merge papers that share a DOI or exceed the configured title similarity threshold."""

    result = deduplicate_papers_with_trail(papers, title_similarity_threshold=title_similarity_threshold)
    return result.unique


def deduplicate_papers_with_trail(
        papers: Iterable[PaperMetadata],
        *,
        title_similarity_threshold: float = 0.92,
) -> DeduplicationResult:
    """Deduplicate with full audit trail and separated duplicates."""

    papers_list = list(papers)
    unique_by_identity: dict[str, PaperMetadata] = {}
    audit: list[DeduplicationAuditEntry] = []
    duplicate_records: list[PaperMetadata] = []

    title_only: list[PaperMetadata] = []

    for paper in papers_list:
        if paper.doi:
            if paper.identity_key in unique_by_identity:
                kept = unique_by_identity[paper.identity_key]
                merged = kept.merge_with(paper)
                unique_by_identity[paper.identity_key] = merged
                duplicate_records.append(paper)
                audit.append(DeduplicationAuditEntry(
                    kept_id=kept.identity_key,
                    removed_id=paper.identity_key,
                    kept_source=kept.source,
                    removed_source=paper.source,
                    method="doi",
                    similarity=1.0,
                    reason=f"DOI match: {paper.doi}",
                ))
            else:
                unique_by_identity[paper.identity_key] = paper
            continue
        title_only.append(paper)

    if title_only:
        texts = [paper.normalized_title for paper in title_only]
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(matrix)

        consumed: set[int] = set()
        for index, paper in enumerate(title_only):
            if index in consumed:
                continue
            merged = paper
            consumed.add(index)
            for candidate_index in range(index + 1, len(title_only)):
                sim = similarities[index, candidate_index]
                if sim >= title_similarity_threshold:
                    candidate = title_only[candidate_index]
                    kept = merged
                    merged = merged.merge_with(candidate)
                    consumed.add(candidate_index)
                    duplicate_records.append(candidate)
                    audit.append(DeduplicationAuditEntry(
                        kept_id=kept.identity_key,
                        removed_id=candidate.identity_key,
                        kept_source=kept.source,
                        removed_source=candidate.source,
                        method="title_similarity",
                        similarity=round(float(sim), 4),
                        reason=f"Title similarity {sim:.4f} ≥ {title_similarity_threshold}",
                    ))
            unique_by_identity[merged.identity_key] = (
                unique_by_identity[merged.identity_key].merge_with(merged)
                if merged.identity_key in unique_by_identity
                else merged
            )

    return DeduplicationResult(
        unique=list(unique_by_identity.values()),
        duplicates=duplicate_records,
        audit_trail=audit,
    )
