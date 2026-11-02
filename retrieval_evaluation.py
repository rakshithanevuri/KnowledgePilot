
"""
KnowledgePilot - Document-Aware Retrieval Evaluation

Generates evaluation questions for uploaded PDFs using Gemini,
then compares:
1. Semantic-only retrieval
2. Keyword-only retrieval (TF-IDF)
3. Hybrid retrieval (Reciprocal Rank Fusion)

Metrics:
1. Hit Rate@K
2. Recall@K
3. Mean Reciprocal Rank (MRR)
"""

import json
import re

import numpy as np
from google.genai import types

from rag_engine import (
    get_gemini_client,
    lexical_similarity_scores,
    retrieve_chunks,
)


# --------------------------------------------------
# NORMALIZATION
# --------------------------------------------------

def normalize_source(source):
    """Normalize filenames for reliable comparison."""
    return (
        str(source)
        .strip()
        .replace("\\", "/")
        .split("/")[-1]
        .casefold()
    )


def normalize_page(page):
    """Normalize page values for comparison."""
    return str(page).strip()


# --------------------------------------------------
# GENERATE DOCUMENT-SPECIFIC EVALUATION QUESTIONS
# --------------------------------------------------

def generate_evaluation_cases(
    chunks,
    api_key,
    model_name="gemini-3.5-flash",
    questions_per_document=3,
):
    """
    Generate questions and reference page labels from uploaded PDFs.

    Uses one Gemini API request.
    The generated reference labels should be treated as AI-generated
    and may require human review.
    """

    if not chunks:
        raise ValueError("No document chunks are available.")

    if not api_key:
        raise ValueError("Gemini API key is required.")

    # Group chunk text by source and page.
    pages = {}

    for chunk in chunks:
        source = chunk.get("source")
        page = chunk.get("page")
        text = chunk.get("text", "")

        if not source or page is None or not text.strip():
            continue

        key = (str(source), str(page))

        if key not in pages:
            pages[key] = []

        pages[key].append(text.strip())

    if not pages:
        raise ValueError("No usable document text was found.")

    # Prepare a compact document representation.
    # Limit the input to avoid sending excessively large PDFs.
    document_sections = []

    for (source, page), texts in pages.items():
        page_text = "\n".join(texts)

        document_sections.append(
            {
                "source": source,
                "page": page,
                "text": page_text[:1800],
            }
        )

    selected_sections = document_sections[:30]

    document_text = ""

    for section in selected_sections:
        document_text += (
            f"\n\nSOURCE: {section['source']}\n"
            f"PAGE: {section['page']}\n"
            f"CONTENT:\n{section['text']}\n"
        )

    prompt = f"""
You are generating a retrieval evaluation dataset for a PDF research assistant.

Read the document passages below and create approximately
{questions_per_document} useful evaluation questions per PDF.

Requirements:
1. Questions must be answerable from the supplied passages.
2. Questions should test meaningful facts, not trivial wording.
3. Provide a concise expected answer for each question.
4. For every question, identify all relevant source filenames and page numbers.
5. Only use source filenames and page numbers supplied below.
6. Do not invent facts or page numbers.
7. Avoid duplicate or nearly identical questions.
8. If a document does not contain enough information, generate fewer questions.

Return valid JSON only, using this structure:

{{
  "evaluation_cases": [
    {{
      "question": "A question based on the document",
      "expected_answer": "A concise answer supported by the document",
      "relevant_passages": [
        {{
          "source": "document.pdf",
          "page": 1
        }}
      ]
    }}
  ]
}}

DOCUMENT PASSAGES:
{document_text}
"""

    client = get_gemini_client(api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    response_text = response.text

    if not response_text:
        raise ValueError("Gemini returned an empty evaluation dataset.")

    # Remove accidental Markdown code fences, if present.
    response_text = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$",
        "",
        response_text.strip(),
        flags=re.IGNORECASE,
    )

    try:
        generated_data = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Could not parse the generated evaluation questions."
        ) from exc

    cases = generated_data.get("evaluation_cases", [])

    if not isinstance(cases, list):
        raise ValueError("Invalid evaluation dataset format.")

    # Only allow labels that refer to pages actually present in the chunks.
    available_pages = {
        (
            normalize_source(chunk.get("source", "")),
            normalize_page(chunk.get("page", "")),
        )
        for chunk in chunks
    }

    validated_cases = []

    for case in cases:
        if not isinstance(case, dict):
            continue

        question = str(case.get("question", "")).strip()
        expected_answer = str(
            case.get("expected_answer", "")
        ).strip()

        relevant_passages = case.get("relevant_passages", [])

        if not question or not isinstance(relevant_passages, list):
            continue

        valid_passages = []

        for passage in relevant_passages:
            if not isinstance(passage, dict):
                continue

            source = str(passage.get("source", "")).strip()
            page = passage.get("page")

            if not source or page is None:
                continue

            try:
                page_number = int(page)
            except (TypeError, ValueError):
                continue

            if (
                normalize_source(source),
                normalize_page(page_number),
            ) in available_pages:
                valid_passages.append(
                    {
                        "source": source,
                        "page": page_number,
                    }
                )

        if not valid_passages:
            continue

        validated_cases.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "relevant_passages": valid_passages,
            }
        )

    if not validated_cases:
        raise ValueError(
            "No valid evaluation questions could be generated "
            "for the selected documents."
        )

    return validated_cases


# --------------------------------------------------
# RELEVANCE CHECK
# --------------------------------------------------

def is_relevant(result, relevant_passages):
    """Check whether a result matches a labelled source/page pair."""

    result_source = normalize_source(result.get("source", ""))
    result_page = normalize_page(result.get("page", ""))

    for passage in relevant_passages:
        expected_source = normalize_source(passage.get("source", ""))
        expected_page = normalize_page(passage.get("page", ""))

        if (
            result_source == expected_source
            and result_page == expected_page
        ):
            return True

    return False


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def hit_rate_at_k(results, relevant_passages):
    """Return 1 if at least one relevant passage is retrieved."""

    if not relevant_passages:
        return None

    return float(
        any(is_relevant(result, relevant_passages) for result in results)
    )


def recall_at_k(results, relevant_passages):
    """Calculate the proportion of labelled relevant passages retrieved."""

    if not relevant_passages:
        return None

    expected = {
        (
            normalize_source(p.get("source", "")),
            normalize_page(p.get("page", "")),
        )
        for p in relevant_passages
    }

    retrieved = {
        (
            normalize_source(r.get("source", "")),
            normalize_page(r.get("page", "")),
        )
        for r in results
    }

    return len(expected.intersection(retrieved)) / len(expected)


def reciprocal_rank(results, relevant_passages):
    """Return reciprocal rank of the first relevant result."""

    if not relevant_passages:
        return None

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, relevant_passages):
            return 1.0 / rank

    return 0.0


# --------------------------------------------------
# RETRIEVAL STRATEGIES
# --------------------------------------------------

def retrieve_semantic_only(question, chunks, retriever, top_k=5):
    """Retrieve chunks using semantic FAISS search only."""

    query_embedding = retriever["model"].encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    scores, indices = retriever["index"].search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, index in zip(scores[0], indices[0]):
        if index < 0:
            continue

        result = dict(chunks[int(index)])
        result["score"] = float(score)
        result["semantic_score"] = float(score)
        result["lexical_score"] = 0.0

        results.append(result)

    return results


def retrieve_keyword_only(question, chunks, retriever, top_k=5):
    """Retrieve chunks using TF-IDF keyword similarity only."""

    scores = lexical_similarity_scores(
        question,
        chunks,
        vectorizer=retriever.get("vectorizer"),
        matrix=retriever.get("lexical_matrix"),
    )

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        if scores[index] <= 0:
            continue

        result = dict(chunks[int(index)])
        result["score"] = float(scores[index])
        result["semantic_score"] = 0.0
        result["lexical_score"] = float(scores[index])

        results.append(result)

    return results


# --------------------------------------------------
# EVALUATE ONE STRATEGY
# --------------------------------------------------

def evaluate_strategy(retrieved, relevant_passages):
    """Calculate retrieval metrics for one test question."""

    return {
        "hit_rate": hit_rate_at_k(retrieved, relevant_passages),
        "recall": recall_at_k(retrieved, relevant_passages),
        "reciprocal_rank": reciprocal_rank(
            retrieved,
            relevant_passages,
        ),
        "retrieved_passages": [
            {
                "source": result.get("source", "Unknown"),
                "page": result.get("page", "Unknown"),
                "score": result.get("score"),
            }
            for result in retrieved
        ],
    }


# --------------------------------------------------
# COMPARE RETRIEVAL STRATEGIES
# --------------------------------------------------

def evaluate_retrieval_strategies(
    test_cases,
    chunks,
    retriever,
    top_k=5,
):
    """
    Compare semantic-only, keyword-only, and hybrid retrieval.

    No Gemini API calls are made by this function.
    """

    if not test_cases:
        raise ValueError("At least one evaluation question is required.")

    if not chunks:
        raise ValueError("No document chunks are available.")

    if retriever is None:
        raise ValueError("The retriever has not been initialized.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    strategies = {
        "Semantic-only": [],
        "Keyword-only": [],
        "Hybrid RRF": [],
    }

    for case in test_cases:
        question = case["question"].strip()
        relevant = case["relevant_passages"]

        if not question:
            continue

        if not relevant:
            continue

        semantic = retrieve_semantic_only(
            question,
            chunks,
            retriever,
            top_k,
        )

        keyword = retrieve_keyword_only(
            question,
            chunks,
            retriever,
            top_k,
        )

        hybrid = retrieve_chunks(
            question=question,
            chunks=chunks,
            retriever=retriever,
            top_k=top_k,
        )

        for name, retrieved in [
            ("Semantic-only", semantic),
            ("Keyword-only", keyword),
            ("Hybrid RRF", hybrid),
        ]:
            result = evaluate_strategy(retrieved, relevant)
            result["question"] = question
            result["expected_answer"] = case.get("expected_answer", "")
            strategies[name].append(result)

    comparison = {}

    for name, cases in strategies.items():
        if not cases:
            continue

        comparison[name] = {
            "hit_rate_at_k": float(
                np.mean([case["hit_rate"] for case in cases])
            ),
            "recall_at_k": float(
                np.mean([case["recall"] for case in cases])
            ),
            "mrr": float(
                np.mean([case["reciprocal_rank"] for case in cases])
            ),
            "individual_results": cases,
        }

    if not comparison:
        raise ValueError("No valid evaluation cases could be evaluated.")

    return comparison
