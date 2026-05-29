"""MeSH term lookup via the NLM E-utilities API."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from config import ResearchConfig

logger = logging.getLogger(__name__)

MESH_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
MESH_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


@dataclass
class MeshTerm:
    descriptor_ui: str = ""
    descriptor_name: str = ""
    entry_terms: list[str] = field(default_factory=list)


def lookup_mesh_terms(
    keywords: list[str],
    *,
    config: ResearchConfig,
    session: object,
    limiter: object,
    timeout: int = 30,
) -> dict[str, MeshTerm]:
    result: dict[str, MeshTerm] = {}
    for keyword in keywords:
        try:
            term = _lookup_single(keyword, session=session, limiter=limiter, timeout=timeout)
            if term.descriptor_name:
                result[keyword] = term
                logger.info("MeSH: '%s' → '%s' (UI: %s)", keyword, term.descriptor_name, term.descriptor_ui)
            else:
                result[keyword] = MeshTerm()
                logger.debug("MeSH: no mapping found for '%s'", keyword)
        except Exception:
            logger.warning("MeSH lookup failed for '%s'", keyword, exc_info=True)
            result[keyword] = MeshTerm()
    return result


def _lookup_single(
    keyword: str,
    *,
    session: object,
    limiter: object,
    timeout: int,
) -> MeshTerm:
    if limiter is not None:
        limiter.wait()

    search_params = {
        "db": "mesh",
        "term": keyword,
        "retmode": "json",
        "retmax": "1",
    }
    resp = session.get(MESH_SEARCH_URL, params=search_params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return MeshTerm()

    mesh_uid = id_list[0]

    if limiter is not None:
        limiter.wait()

    fetch_params = {
        "db": "mesh",
        "id": mesh_uid,
        "rettype": "full",
    }
    resp2 = session.get(MESH_FETCH_URL, params=fetch_params, timeout=timeout)
    resp2.raise_for_status()

    return _parse_mesh_xml(resp2.text, mesh_uid)


def _parse_mesh_xml(xml_text: str, mesh_uid: str) -> MeshTerm:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return MeshTerm()

    descriptor_el = root.find(".//DescriptorRecord/DescriptorName/String")
    descriptor_name = descriptor_el.text.strip() if descriptor_el is not None and descriptor_el.text else ""

    entry_terms: list[str] = []
    for el in root.findall(".//DescriptorRecord/ConceptList/Concept/TermList/Term/String"):
        if el.text and el.text.strip() != descriptor_name:
            entry_terms.append(el.text.strip())

    return MeshTerm(
        descriptor_ui=mesh_uid,
        descriptor_name=descriptor_name,
        entry_terms=entry_terms[:10],
    )
