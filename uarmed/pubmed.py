"""OPTIONAL: search live PubMed for questions outside the bundled dataset.

The bundled PubMedQA file has only 1,000 abstracts, so most everyday questions
("What is melatonin?") are simply not covered. NCBI offers a free public search
service called E-utilities (no account, no key, no cost; limit about
3 requests per second). We use two of its endpoints:
    esearch -> give a query, get back a list of PubMed IDs
    efetch  -> give PubMed IDs, get back titles and abstracts (as XML)
Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import json
import urllib.parse  # builds "?key=value&..." strings safely
import urllib.request  # sends the web requests
import xml.etree.ElementTree as ET  # reads the XML that efetch returns

from .explainer import split_sentences

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(query, max_results=20, timeout=15):
    """Return a list of articles: {"pmid", "title", "contexts"} (abstract parts)."""
    # Step 1: esearch -> PubMed IDs, best matches first.
    params = urllib.parse.urlencode(
        {"db": "pubmed", "term": query, "retmax": max_results,
         "retmode": "json", "sort": "relevance", "tool": "uar-med"}
    )
    with urllib.request.urlopen(f"{EUTILS}/esearch.fcgi?{params}", timeout=timeout) as response:
        ids = json.load(response)["esearchresult"]["idlist"]
    if not ids:
        return []

    # Step 2: efetch -> the abstracts of those IDs.
    params = urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "tool": "uar-med"}
    )
    with urllib.request.urlopen(f"{EUTILS}/efetch.fcgi?{params}", timeout=timeout) as response:
        return parse_pubmed_xml(response.read())


def parse_pubmed_xml(xml_text):
    """Pull PMID, title and abstract sections out of efetch's XML."""
    root = ET.fromstring(xml_text)
    articles = []
    for article in root.iter("PubmedArticle"):  # one element per paper
        pmid = article.findtext(".//PMID")
        # itertext() also collects text inside tags like <i>...</i>.
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""

        contexts = []
        for section in article.iter("AbstractText"):  # BACKGROUND, METHODS, ...
            text = "".join(section.itertext()).strip()
            if text:
                contexts.extend(chunk_text(text))
        if contexts:  # skip papers without an abstract
            articles.append({"pmid": pmid, "title": title, "contexts": contexts})
    return articles


def chunk_text(text, sentences_per_chunk=4):
    """Many abstracts are one long block. Cut them into passages of a few
    sentences, so retrieval can pick the useful part, like in PubMedQA."""
    sentences = split_sentences(text)
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]
