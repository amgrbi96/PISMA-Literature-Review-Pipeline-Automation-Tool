"""Backward and forward citation expansion over the seeded paper set."""

from __future__ import annotations

from tqdm import tqdm

from config import ResearchConfig
from database import DatabaseManager
from discovery.protocols import CitationProviderProtocol
from models.paper import PaperMetadata
from utils.deduplication import deduplicate_papers


class CitationExpander:
    """Expand the current review set through reference and citation lookups."""

    def __init__(
            self,
            config: ResearchConfig,
            database: DatabaseManager,
            citation_provider: CitationProviderProtocol,
    ) -> None:
        self.config = config
        self.database = database
        self.citation_provider = citation_provider

    def expand(self, papers: list[PaperMetadata]) -> list[PaperMetadata]:
        """Return newly discovered papers found through iterative backward and forward snowballing."""

        if not self.config.citation_snowballing_enabled:
            return []

        depth = self.config.snowballing_depth
        per_direction_limit = self.config.snowballing_per_direction_limit

        seed_limit = min(len(papers), max(5, self.config.max_papers_to_analyze // 2))
        ranked = sorted(papers, key=lambda paper: (paper.citation_count, paper.year or 0), reverse=True)
        seeds = ranked[:seed_limit]

        all_discovered: list[PaperMetadata] = []
        seen_identity_keys: set[str] = {p.identity_key for p in papers}

        current_seeds = seeds
        for iteration in range(depth):
            iteration_discovered: list[PaperMetadata] = []
            for seed in tqdm(
                    current_seeds,
                    desc=f"Citation expansion (depth {iteration + 1}/{depth})",
                    unit="paper",
                    disable=self.config.disable_progress_bars,
            ):
                backward = self.citation_provider.fetch_references(seed, limit=per_direction_limit)
                forward = self.citation_provider.fetch_citations(seed, limit=per_direction_limit)
                references = [paper.citation_label for paper in backward]
                citations = [paper.citation_label for paper in forward]
                if seed.database_id is not None:
                    self.database.update_citations(seed.database_id, references, citations)
                for paper in [*backward, *forward]:
                    if paper.identity_key not in seen_identity_keys:
                        iteration_discovered.append(
                            paper.model_copy(update={"query_key": self.config.query_key})
                        )
                        seen_identity_keys.add(paper.identity_key)

            all_discovered.extend(iteration_discovered)

            if not iteration_discovered:
                break

            current_seeds = iteration_discovered

        return deduplicate_papers(
            all_discovered,
            title_similarity_threshold=self.config.title_similarity_threshold,
        )
