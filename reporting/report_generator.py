"""Report and artifact generation for ranked papers, PRISMA summaries, and exports."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from sqlalchemy import create_engine

from analysis.ai_screener import AIScreener
from config import ResearchConfig
from models.paper import PaperMetadata
from utils.text_processing import top_terms


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{label} ({count})" for label, count in list(counts.items())[:5]) or "none"


def _count_values(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _collect_pass_names(papers: list[PaperMetadata]) -> list[str]:
    pass_names: set[str] = set()
    for paper in papers:
        pass_names.update(paper.screening_details.get("passes", {}).keys())
    return sorted(pass_names)


def _paper_to_dict(paper: PaperMetadata, pass_names: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "database_id": paper.database_id,
        "title": paper.title,
        "authors": "; ".join(paper.authors),
        "abstract": paper.abstract,
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "source": paper.source,
        "citation_count": paper.citation_count,
        "reference_count": paper.reference_count,
        "pdf_link": paper.pdf_link,
        "pdf_path": paper.pdf_path,
        "open_access": paper.open_access,
        "relevance_score": paper.relevance_score,
        "relevance_explanation": paper.relevance_explanation,
        "topic_prefilter_score": paper.screening_details.get("topic_prefilter_score"),
        "topic_prefilter_similarity": paper.screening_details.get("topic_prefilter_similarity"),
        "topic_prefilter_model": paper.screening_details.get("topic_prefilter_model"),
        "topic_prefilter_threshold": paper.screening_details.get("topic_prefilter_threshold"),
        "topic_prefilter_label": paper.screening_details.get("topic_prefilter_label"),
        "topic_prefilter_keyword_overlap": paper.screening_details.get("topic_prefilter_keyword_overlap"),
        "topic_prefilter_research_fit_label": paper.screening_details.get("topic_prefilter_research_fit_label"),
        "topic_prefilter_weighted_score": paper.screening_details.get("topic_prefilter_weighted_score"),
        "topic_prefilter_min_keyword_matches": paper.screening_details.get("topic_prefilter_min_keyword_matches"),
        "topic_prefilter_matched_keyword_count": paper.screening_details.get("topic_prefilter_matched_keyword_count"),
        "topic_prefilter_keyword_rule_count": paper.screening_details.get("topic_prefilter_keyword_rule_count"),
        "topic_prefilter_extracted_topics": json.dumps(
            paper.screening_details.get("topic_prefilter_extracted_topics", []),
            ensure_ascii=False,
        ),
        "topic_prefilter_keyword_details": json.dumps(
            paper.screening_details.get("topic_prefilter_keyword_details", []),
            ensure_ascii=False,
        ),
        "inclusion_decision": paper.inclusion_decision,
        "retain_reason": paper.screening_details.get("retain_reason", ""),
        "exclusion_reason": paper.screening_details.get("exclusion_reason", ""),
        "exclusion_code": paper.screening_details.get("exclusion_code", ""),
        "matched_inclusion_criteria": json.dumps(
            paper.screening_details.get("matched_inclusion_criteria", []),
            ensure_ascii=False,
        ),
        "matched_exclusion_criteria": json.dumps(
            paper.screening_details.get("matched_exclusion_criteria", []),
            ensure_ascii=False,
        ),
        "matched_banned_topics": json.dumps(
            paper.screening_details.get("matched_banned_topics", []),
            ensure_ascii=False,
        ),
        "matched_excluded_title_terms": json.dumps(
            paper.screening_details.get("matched_excluded_title_terms", []),
            ensure_ascii=False,
        ),
        "references": json.dumps(paper.references, ensure_ascii=False),
        "citations": json.dumps(paper.citations, ensure_ascii=False),
        "extracted_passage": paper.extracted_passage,
        "methodology_category": paper.methodology_category,
        "domain_category": paper.domain_category,
    }
    passes = paper.screening_details.get("passes", {})
    for pass_name in pass_names or []:
        pass_payload = passes.get(pass_name, {})
        payload[f"pass_{pass_name}_score"] = pass_payload.get("relevance_score")
        payload[f"pass_{pass_name}_decision"] = pass_payload.get("decision")
        payload[f"pass_{pass_name}_reason"] = (
                pass_payload.get("skip_reason")
                or pass_payload.get("retain_reason")
                or pass_payload.get("exclusion_reason")
                or pass_payload.get("exclusion_code")
                or pass_payload.get("explanation")
                or ""
        )
        payload[f"pass_{pass_name}_provider"] = pass_payload.get("llm_provider", "")
        payload[f"pass_{pass_name}_model"] = pass_payload.get("model_name", "")
        payload[f"pass_{pass_name}_threshold"] = pass_payload.get("threshold")
        payload[f"pass_{pass_name}_min_input_score"] = pass_payload.get("min_input_score")
        payload[f"pass_{pass_name}_skipped"] = pass_payload.get("skipped", False)
    return payload


def _paper_to_dict_keys(pass_names: list[str] | None = None) -> list[str]:
    keys = [
        "database_id",
        "title",
        "authors",
        "abstract",
        "year",
        "venue",
        "doi",
        "source",
        "citation_count",
        "reference_count",
        "pdf_link",
        "pdf_path",
        "open_access",
        "relevance_score",
        "relevance_explanation",
        "topic_prefilter_score",
        "topic_prefilter_similarity",
        "topic_prefilter_model",
        "topic_prefilter_threshold",
        "topic_prefilter_label",
        "topic_prefilter_keyword_overlap",
        "topic_prefilter_research_fit_label",
        "topic_prefilter_weighted_score",
        "topic_prefilter_min_keyword_matches",
        "topic_prefilter_matched_keyword_count",
        "topic_prefilter_keyword_rule_count",
        "topic_prefilter_extracted_topics",
        "topic_prefilter_keyword_details",
        "inclusion_decision",
        "retain_reason",
        "exclusion_reason",
        "exclusion_code",
        "matched_inclusion_criteria",
        "matched_exclusion_criteria",
        "matched_banned_topics",
        "matched_excluded_title_terms",
        "references",
        "citations",
        "extracted_passage",
        "methodology_category",
        "domain_category",
    ]
    for pass_name in pass_names or []:
        keys.extend(
            [
                f"pass_{pass_name}_score",
                f"pass_{pass_name}_decision",
                f"pass_{pass_name}_reason",
                f"pass_{pass_name}_provider",
                f"pass_{pass_name}_model",
                f"pass_{pass_name}_threshold",
                f"pass_{pass_name}_min_input_score",
                f"pass_{pass_name}_skipped",
            ]
        )
    return keys


def _artifact_fingerprint_path(path: Path) -> Path:
    """Return the sidecar file used to track the latest artifact fingerprint."""

    return path.with_suffix(path.suffix + ".sha256")


def _dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Build a stable fingerprint for incremental artifact comparisons."""

    payload = dataframe.to_json(orient="records", date_format="iso", force_ascii=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _rank_papers(papers: list[PaperMetadata]) -> list[PaperMetadata]:
    """Sort papers so exports and summaries emphasize the strongest candidates first."""

    return sorted(
        papers,
        key=lambda paper: (
            paper.relevance_score if paper.relevance_score is not None else -1.0,
            paper.citation_count,
            paper.year or 0,
        ),
        reverse=True,
    )


def _papers_to_dataframe(papers: list[PaperMetadata]) -> pd.DataFrame:
    pass_names = _collect_pass_names(papers)
    if not papers:
        return pd.DataFrame(columns=_paper_to_dict_keys(pass_names))
    return pd.DataFrame([_paper_to_dict(paper, pass_names) for paper in papers])


def _write_artifact_fingerprint(path: Path, fingerprint: str) -> None:
    """Persist the current fingerprint so incremental runs can skip unchanged rewrites."""

    _artifact_fingerprint_path(path).write_text(fingerprint, encoding="utf-8")


def _read_artifact_fingerprint(path: Path) -> str | None:
    """Load a stored artifact fingerprint when the sidecar file exists."""

    fingerprint_path = _artifact_fingerprint_path(path)
    if not fingerprint_path.exists():
        return None
    try:
        return fingerprint_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


class ReportGenerator:
    """Render the pipeline state into CSV, JSON, Markdown, and SQLite outputs."""

    def __init__(self, config: ResearchConfig, ai_screener: AIScreener) -> None:
        self.config = config
        self.ai_screener = ai_screener

    def generate(self, papers: list[PaperMetadata], *, stats: dict[str, Any] | None = None) -> dict[str, str]:
        """Generate all configured artifacts and return a mapping of logical names to file paths."""

        self._clear_previous_outputs()
        ranked = _rank_papers(papers)
        scored = [paper for paper in ranked if paper.relevance_score is not None]
        final_threshold = self._final_threshold()
        shortlisted = [
            paper
            for paper in scored
            if (paper.relevance_score or 0.0) >= final_threshold
               and paper.inclusion_decision != "exclude"
        ]
        excluded = [paper for paper in scored if paper.inclusion_decision == "exclude"]
        outputs: dict[str, str] = {}

        if self.config.output_csv:
            csv_path = self._write_csv(ranked)
            included_csv_path = self._write_decision_csv("included_papers.csv", shortlisted)
            excluded_csv_path = self._write_decision_csv("excluded_papers.csv", excluded)
            outputs.update(
                {
                    "papers_csv": str(csv_path),
                    "included_papers_csv": str(included_csv_path),
                    "excluded_papers_csv": str(excluded_csv_path),
                }
            )

        graph_path: Path | None = None
        if self.config.output_json:
            top_json_path = self._write_top_papers_json(shortlisted or scored[:20] or ranked[:20])
            graph_path = self._write_citation_graph(ranked)
            prisma_flow_json_path = self._write_prisma_flow_json(ranked, shortlisted, excluded, stats or {})
            outputs.update(
                {
                    "top_papers_json": str(top_json_path),
                    "citation_graph_json": str(graph_path),
                    "prisma_flow_json": str(prisma_flow_json_path),
                }
            )

        if self.config.output_markdown:
            prisma_flow_md_path = self._write_prisma_flow_md(ranked, shortlisted, excluded, stats or {})
            summary_path = self._write_review_summary(ranked, shortlisted, graph_path)
            outputs.update(
                {
                    "prisma_flow_md": str(prisma_flow_md_path),
                    "review_summary_md": str(summary_path),
                }
            )

        if self.config.output_sqlite_exports:
            included_db_path = self._write_decision_database("included_papers.db", "included_papers", shortlisted)
            excluded_db_path = self._write_decision_database("excluded_papers.db", "excluded_papers", excluded)
            outputs.update(
                {
                    "included_papers_db": str(included_db_path),
                    "excluded_papers_db": str(excluded_db_path),
                }
            )

        if self.config.output_ris:
            ris_path = self._write_ris_export(ranked)
            outputs["ris_export"] = str(ris_path)

        if self.config.output_bibtex:
            bibtex_path = self._write_bibtex_export(ranked)
            outputs["bibtex_export"] = str(bibtex_path)

        strategy_md_path = self._write_search_strategy_md(stats or {})
        strategy_json_path = self._write_search_strategy_json(stats or {})
        outputs["search_strategy_md"] = str(strategy_md_path)
        outputs["search_strategy_json"] = str(strategy_json_path)

        mermaid_path = self._write_prisma_flow_mermaid(ranked, shortlisted, excluded, stats or {})
        outputs["prisma_flow_mermaid"] = str(mermaid_path)

        quality_report = stats.get("metadata_quality_report")
        if quality_report:
            quality_path = self._write_metadata_quality_json(quality_report)
            outputs["metadata_quality_json"] = str(quality_path)

        return outputs

    def _clear_previous_outputs(self) -> None:
        """Remove stale artifacts so each run produces a self-consistent result directory."""

        if self.config.incremental_report_regeneration:
            return
        for filename in (
                "papers.csv",
                "included_papers.csv",
                "excluded_papers.csv",
                "top_papers.json",
                "citation_graph.json",
                "prisma_flow.json",
                "prisma_flow.md",
                "included_papers.db",
                "excluded_papers.db",
                "review_summary.md",
                "papers.ris",
                "papers.bib",
                "search_strategy.md",
                "search_strategy.json",
                "prisma_flow.mermaid",
                "metadata_quality.json",
        ):
            path = Path(self.config.results_dir) / filename
            if path.exists():
                path.unlink()
            fingerprint_path = _artifact_fingerprint_path(path)
            if fingerprint_path.exists():
                fingerprint_path.unlink()

    def _write_csv(self, papers: list[PaperMetadata]) -> Path:
        path = Path(self.config.results_dir) / "papers.csv"
        dataframe = _papers_to_dataframe(papers)
        self._write_dataframe_csv(path, dataframe)
        return path

    def _write_top_papers_json(self, papers: list[PaperMetadata]) -> Path:
        path = Path(self.config.results_dir) / "top_papers.json"
        pass_names = _collect_pass_names(papers)
        payload = [_paper_to_dict(paper, pass_names) for paper in papers[:25]]
        self._write_json_artifact(path, payload)
        return path

    def _write_decision_csv(self, filename: str, papers: list[PaperMetadata]) -> Path:
        path = Path(self.config.results_dir) / filename
        self._write_dataframe_csv(path, _papers_to_dataframe(papers))
        return path

    def _write_citation_graph(self, papers: list[PaperMetadata]) -> Path:
        path = Path(self.config.results_dir) / "citation_graph.json"
        graph = nx.DiGraph()
        for paper in papers:
            node_id = paper.citation_label
            graph.add_node(
                node_id,
                title=paper.title,
                score=paper.relevance_score,
                decision=paper.inclusion_decision,
            )
            for reference in paper.references:
                if reference:
                    graph.add_node(reference, title=reference)
                    graph.add_edge(node_id, reference)
            for citation in paper.citations:
                if citation:
                    graph.add_node(citation, title=citation)
                    graph.add_edge(citation, node_id)

        payload = {
            "nodes": [{"id": node, **attrs} for node, attrs in graph.nodes(data=True)],
            "edges": [{"source": source, "target": target} for source, target in graph.edges()],
            "graph_metrics": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
            },
        }
        self._write_json_artifact(path, payload)
        return path

    def _write_review_summary(
            self,
            ranked: list[PaperMetadata],
            shortlisted: list[PaperMetadata],
            graph_path: Path | None,
    ) -> Path:
        """Create the final narrative summary using the configured LLM or a heuristic fallback."""

        path = Path(self.config.results_dir) / "review_summary.md"
        llm_summary = None if self.config.run_mode == "collect" else self.ai_screener.summarize_review(shortlisted or ranked[:12])
        if llm_summary:
            body = llm_summary.strip()
        else:
            body = self._heuristic_summary(ranked, shortlisted)

        body += "\n\n## Outputs\n"
        if self.config.output_csv:
            body += f"- CSV summary: `{Path(self.config.results_dir) / 'papers.csv'}`\n"
            body += f"- Included papers CSV: `{Path(self.config.results_dir) / 'included_papers.csv'}`\n"
            body += f"- Excluded papers CSV: `{Path(self.config.results_dir) / 'excluded_papers.csv'}`\n"
        if self.config.output_json:
            body += f"- Ranked shortlist JSON: `{Path(self.config.results_dir) / 'top_papers.json'}`\n"
            if graph_path is not None:
                body += f"- Citation graph data: `{graph_path}`\n"
        if self.config.output_sqlite_exports:
            body += f"- Included papers DB: `{Path(self.config.results_dir) / 'included_papers.db'}`\n"
            body += f"- Excluded papers DB: `{Path(self.config.results_dir) / 'excluded_papers.db'}`\n"
        self._write_text_artifact(path, body)
        return path

    def _write_prisma_flow_json(
            self,
            ranked: list[PaperMetadata],
            included: list[PaperMetadata],
            excluded: list[PaperMetadata],
            stats: dict[str, Any],
    ) -> Path:
        path = Path(self.config.results_dir) / "prisma_flow.json"
        decision_counts = stats.get("decision_counts", {})
        payload = {
            "identification": {
                "records_identified": stats.get("discovered_count", len(ranked)),
                "records_after_duplicates_removed": stats.get("deduplicated_count", len(ranked)),
                "records_added_via_snowballing": stats.get("snowballing_added_count", 0),
            },
            "screening": {
                "records_screened": stats.get("screened_count", len([paper for paper in ranked if paper.inclusion_decision])),
                "records_excluded": len(excluded),
                "records_marked_maybe": decision_counts.get("maybe", 0),
                "full_text_records_screened": stats.get("full_text_screened_count", 0),
            },
            "included": {
                "studies_included": len(included),
            },
            "thresholds": {
                "relevance_threshold": self._final_threshold(),
                "decision_mode": self.config.resolved_analysis_passes[-1].decision_mode
                if self.config.resolved_analysis_passes
                else self.config.decision_mode,
                "banned_topics": self.config.banned_topics,
                "run_mode": self.config.run_mode,
            },
        }
        self._write_json_artifact(path, payload)
        return path

    def _write_prisma_flow_md(
            self,
            ranked: list[PaperMetadata],
            included: list[PaperMetadata],
            excluded: list[PaperMetadata],
            stats: dict[str, Any],
    ) -> Path:
        path = Path(self.config.results_dir) / "prisma_flow.md"
        decision_counts = stats.get("decision_counts", {})
        lines = [
            "# PRISMA Flow Summary",
            "",
            "## Identification",
            f"- Records identified: {stats.get('discovered_count', len(ranked))}",
            f"- Records after deduplication: {stats.get('deduplicated_count', len(ranked))}",
            f"- Records added via snowballing: {stats.get('snowballing_added_count', 0)}",
            "",
            "## Screening",
            f"- Records screened: {stats.get('screened_count', len([paper for paper in ranked if paper.inclusion_decision]))}",
            f"- Records excluded: {len(excluded)}",
            f"- Records marked maybe: {decision_counts.get('maybe', 0)}",
            f"- Full-text records screened: {stats.get('full_text_screened_count', 0)}",
            "",
            "## Included",
            f"- Studies included: {len(included)}",
        ]
        self._write_text_artifact(path, "\n".join(lines))
        return path

    def _write_prisma_flow_mermaid(
            self,
            ranked: list[PaperMetadata],
            included: list[PaperMetadata],
            excluded: list[PaperMetadata],
            stats: dict[str, Any],
    ) -> Path:
        path = Path(self.config.results_dir) / "prisma_flow.mermaid"
        decision_counts = stats.get("decision_counts", {})
        identified = stats.get("discovered_count", len(ranked))
        after_dedup = stats.get("deduplicated_count", len(ranked))
        snowballed = stats.get("snowballing_added_count", 0)
        screened = stats.get("screened_count", len([paper for paper in ranked if paper.inclusion_decision]))
        n_excluded = len(excluded)
        n_maybe = decision_counts.get("maybe", 0)
        n_included = len(included)

        lines = [
            "flowchart TD",
            f'    A["Records identified (n={identified})"]',
            f'    B["After deduplication (n={after_dedup})"]',
            f'    C["Added via snowballing (n={snowballed})"]',
            f'    D["Records screened (n={screened})"]',
            f'    E["Excluded (n={n_excluded})"]',
            f'    F["Maybe (n={n_maybe})"]',
            f'    G["Included (n={n_included})"]',
            "",
            "    A --> B",
            "    B --> D",
            "    C --> D",
            "    D --> E",
            "    D --> F",
            "    D --> G",
            "",
        ]
        self._write_text_artifact(path, "\n".join(lines))
        return path

    def _write_metadata_quality_json(self, report: dict[str, Any]) -> Path:
        path = Path(self.config.results_dir) / "metadata_quality.json"
        self._write_json_artifact(path, report)
        return path

    def _write_decision_database(self, filename: str, table_name: str, papers: list[PaperMetadata]) -> Path:
        path = Path(self.config.results_dir) / filename
        dataframe = _papers_to_dataframe(papers)
        fingerprint = _dataframe_fingerprint(dataframe)
        if self.config.incremental_report_regeneration and path.exists():
            if _read_artifact_fingerprint(path) == fingerprint:
                return path
        engine = create_engine(f"sqlite:///{path}")
        try:
            with engine.begin() as connection:
                dataframe.to_sql(table_name, connection, if_exists="replace", index=False)
        finally:
            engine.dispose()
        _write_artifact_fingerprint(path, fingerprint)
        return path

    def _write_dataframe_csv(self, path: Path, dataframe: pd.DataFrame) -> None:
        """Write a CSV artifact, skipping the rewrite when incremental mode sees no change."""

        csv_text = dataframe.to_csv(index=False)
        self._write_text_artifact(path, csv_text)

    def _write_json_artifact(self, path: Path, payload: Any) -> None:
        """Write a JSON artifact, skipping the rewrite when incremental mode sees no change."""

        self._write_text_artifact(path, json.dumps(payload, indent=2, ensure_ascii=False))

    def _write_text_artifact(self, path: Path, content: str) -> None:
        """Write a text artifact, optionally skipping unchanged content in incremental mode."""

        normalized_content = content.replace("\r\n", "\n")
        if self.config.incremental_report_regeneration and path.exists():
            try:
                if path.read_text(encoding="utf-8").replace("\r\n", "\n") == normalized_content:
                    return
            except OSError:
                pass
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(normalized_content)

    def _heuristic_summary(self, ranked: list[PaperMetadata], shortlisted: list[PaperMetadata]) -> str:
        """Create a compact summary when no external LLM summary is available."""

        scored = [paper for paper in ranked if paper.relevance_score is not None]
        candidate_set = shortlisted or scored[:10] or ranked[:10]
        top_keywords = top_terms([f"{paper.title} {paper.abstract}" for paper in candidate_set], limit=10)
        method_counts = _count_values([paper.methodology_category or "unspecified" for paper in candidate_set])
        domain_counts = _count_values([paper.domain_category or "general" for paper in candidate_set])

        lines = [
            "# Literature Review Summary",
            "",
            "## Theme Overview",
            f"The strongest papers cluster around: {', '.join(top_keywords[:6]) or 'no stable term cluster detected'}.",
        ]
        if self.config.run_mode == "collect":
            lines.extend(
                [
                    "",
                    "## Run Mode",
                    "This run collected and merged metadata only. AI screening was not executed.",
                ]
            )
        lines.extend(
            [
                "",
                "## Methods",
                f"Most common methodology labels: {_format_counts(method_counts)}.",
                "",
                "## Domains",
                f"Most common domain labels: {_format_counts(domain_counts)}.",
                "",
                "## Gaps",
                "The shortlist still contains papers with limited methodological specificity or weak theoretical framing. "
                "Those are candidates for manual full-text review before final inclusion.",
                "",
                "## Recommended Core Papers",
            ]
        )
        for paper in candidate_set[:10]:
            lines.append(
                f"- {paper.title} ({paper.year or 'n.d.'}, {paper.venue or 'Unknown venue'}) "
                f"- score {paper.relevance_score or 0:.1f}, decision {paper.inclusion_decision or 'unreviewed'}."
            )
        return "\n".join(lines)

    def _write_search_strategy_md(self, stats: dict[str, Any]) -> Path:
        path = Path(self.config.results_dir) / "search_strategy.md"
        source_records = stats.get("source_query_records", [])
        lines = [
            "# Search Strategy",
            "",
            "## Search Parameters",
            f"- **Research topic**: {self.config.research_topic}",
            f"- **Keywords**: {', '.join(self.config.search_keywords)}",
            f"- **Boolean operator**: {self.config.boolean_operators}",
            f"- **Year range**: {self.config.year_range_start}–{self.config.year_range_end}",
            f"- **Discovery strategy**: {self.config.discovery_strategy}",
            f"- **MeSH expansion**: {'enabled' if self.config.effective_mesh_expansion_enabled else 'disabled'}",
            f"- **Citation snowballing**: depth {self.config.snowballing_depth}, limit {self.config.snowballing_per_direction_limit}/direction",
            "",
            "## Query Variants",
            "",
        ]
        for query in self.config.discovery_queries:
            lines.append(f"- `{query}`")
        lines.append("")
        if source_records:
            lines.append("## Source Results")
            lines.append("")
            lines.append("| Source | Started (UTC) | Duration (s) | Results |")
            lines.append("|--------|---------------|-------------|---------|")
            for rec in source_records:
                lines.append(
                    f"| {rec['source']} | {rec['started_at'][:19]} | {rec['duration_seconds']:.1f} | {rec['results_returned']} |"
                )
            lines.append("")
        lines.append("## Deduplication")
        lines.append(
            f"- Records before deduplication: {stats.get('discovered_count', 'N/A')}"
        )
        lines.append(
            f"- Records after deduplication: {stats.get('deduplicated_count', 'N/A')}"
        )
        lines.append(
            f"- Title similarity threshold: {self.config.title_similarity_threshold}"
        )
        lines.append(
            f"- Records added via snowballing: {stats.get('snowballing_added_count', 0)}"
        )
        self._write_text_artifact(path, "\n".join(lines))
        return path

    def _write_search_strategy_json(self, stats: dict[str, Any]) -> Path:
        path = Path(self.config.results_dir) / "search_strategy.json"
        payload = {
            "search_parameters": {
                "research_topic": self.config.research_topic,
                "keywords": self.config.search_keywords,
                "boolean_operator": self.config.boolean_operators,
                "year_range": [self.config.year_range_start, self.config.year_range_end],
                "discovery_strategy": self.config.discovery_strategy,
                "mesh_expansion_enabled": self.config.effective_mesh_expansion_enabled,
                "snowballing_depth": self.config.snowballing_depth,
                "snowballing_per_direction_limit": self.config.snowballing_per_direction_limit,
            },
            "query_variants": list(self.config.discovery_queries),
            "source_results": stats.get("source_query_records", []),
            "deduplication": {
                "records_before": stats.get("discovered_count"),
                "records_after": stats.get("deduplicated_count"),
                "title_similarity_threshold": self.config.title_similarity_threshold,
                "snowballing_added": stats.get("snowballing_added_count", 0),
            },
        }
        self._write_json_artifact(path, payload)
        return path

    def _write_ris_export(self, papers: list[PaperMetadata], filename: str = "papers.ris") -> Path:
        path = Path(self.config.results_dir) / filename
        entries: list[str] = []
        for paper in papers:
            lines = ["TY  - JOUR"]
            if paper.title:
                lines.append(f"TI  - {paper.title}")
            for author in paper.authors or []:
                lines.append(f"AU  - {author}")
            if paper.year:
                lines.append(f"PY  - {paper.year}")
            if paper.venue:
                lines.append(f"JO  - {paper.venue}")
            if paper.doi:
                lines.append(f"DO  - {paper.doi}")
            if paper.abstract:
                lines.append(f"AB  - {paper.abstract}")
            if paper.external_ids and paper.external_ids.get("pubmed"):
                lines.append(f"ID  - PMID:{paper.external_ids['pubmed']}")
            lines.append("ER  - ")
            entries.append("\n".join(lines))
        self._write_text_artifact(path, "\n\n".join(entries))
        return path

    def _write_bibtex_export(self, papers: list[PaperMetadata], filename: str = "papers.bib") -> Path:
        path = Path(self.config.results_dir) / filename
        entries: list[str] = []
        for paper in papers:
            cite_key = self._bibtex_cite_key(paper)
            lines = [f"@article{{{cite_key},"]
            if paper.title:
                lines.append(f"  title = {{{paper.title}}},")
            if paper.authors:
                lines.append(f"  author = {{{' and '.join(paper.authors)}}},")
            if paper.year:
                lines.append(f"  year = {{{paper.year}}},")
            if paper.venue:
                lines.append(f"  journal = {{{paper.venue}}},")
            if paper.doi:
                lines.append(f"  doi = {{{paper.doi}}},")
            if paper.abstract:
                lines.append(f"  abstract = {{{paper.abstract}}},")
            lines.append("}")
            entries.append("\n".join(lines))
        self._write_text_artifact(path, "\n\n".join(entries))
        return path

    def _bibtex_cite_key(self, paper: PaperMetadata) -> str:
        first_author = paper.authors[0].split()[-1].lower() if paper.authors else "unknown"
        year = paper.year or "nd"
        title_word = paper.title.split()[0].lower() if paper.title else "untitled"
        return f"{first_author}{year}{title_word}"

    def _final_threshold(self) -> float:
        resolved_passes = self.config.resolved_analysis_passes
        if resolved_passes:
            return resolved_passes[-1].threshold
        return self.config.relevance_threshold
