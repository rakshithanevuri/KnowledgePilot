
import streamlit as st
import hashlib
import fitz
from datetime import datetime
import html

from rag_engine import (
    extract_pdf_pages,
    split_into_chunks,
    build_retriever,
    retrieve_chunks,
    generate_with_gemini,
    generate_document_overview,
    generate_learning_material,
    classify_gemini_error,
)
from retrieval_evaluation import (
    generate_evaluation_cases,
    evaluate_retrieval_strategies,
)
from answer_evaluation import (
    validate_citations,
    evaluate_answer,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KnowledgePilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }

    /* Main content width */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    /* This version uses a top navigation bar instead of a sidebar. */
    section[data-testid="stSidebar"] { display: none; }
    button[kind="header"] { display: none; }

    .topbar {
        background: linear-gradient(135deg, #151D31 0%, #111827 100%);
        border: 1px solid #2B3852;
        border-radius: 16px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }

    .hero-panel {
        background: radial-gradient(circle at top right, #29265A 0%, #151D31 42%, #111827 100%);
        border: 1px solid #34415E;
        border-radius: 20px;
        padding: 30px;
        margin: 8px 0 22px 0;
    }

    .feature-card {
        min-height: 155px;
        background: #131A2A;
        border: 1px solid #273449;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 10px;
    }

    .answer-panel {
        background: #131A2A;
        border: 1px solid #334155;
        border-left: 4px solid #8B5CF6;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 8px 0 16px 0;
        line-height: 1.75;
    }

    @media (max-width: 800px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero-panel { padding: 20px; }
    }

    /* Headings */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 650 !important;
    }

    /* Brand */
    .brand {
        font-size: 25px;
        font-weight: 800;
        color: #A78BFA;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .brand-caption {
        color: #94A3B8;
        font-size: 12px;
        margin-bottom: 25px;
    }

    /* Page heading */
    .page-title {
        font-size: 32px;
        font-weight: 750;
        color: #F8FAFC;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .page-subtitle {
        color: #94A3B8;
        font-size: 14px;
        margin-bottom: 28px;
    }

    /* Cards */
    .custom-card {
        background: #131A2A;
        border: 1px solid #273449;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 30px;
        font-weight: 750;
    }

    .document-title {
        color: #F8FAFC;
        font-size: 16px;
        font-weight: 650;
        overflow-wrap: anywhere;
    }

    .muted-text {
        color: #94A3B8;
        font-size: 13px;
    }

    .section-label {
        color: #A78BFA;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        border: 1px solid #6D5AE6;
        background: #6D5AE6;
        color: white;
        font-weight: 600;
        min-height: 40px;
    }

    .stButton > button:hover {
        background: #806CF0;
        border-color: #806CF0;
        color: white;
    }

    /* Inputs */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #131A2A !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 9px !important;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #131A2A;
        border-color: #334155;
    }

    /* Dividers */
    hr {
        border-color: #263247;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background-color: #131A2A;
        border: 1px solid #273449;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
        line-height: 1.75;
    }

    [data-testid="stMarkdownContainer"] p { line-height: 1.72; }
    [data-testid="stMarkdownContainer"] li { margin-bottom: 0.35rem; }
    [data-testid="stAlert"] { border-radius: 12px; }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "documents": {},
    "chunks": {},
    "overviews": {},
    "retrieval_cache": {},
    "chat_histories": {},
    "selected_document_ids": [],
    "active_view": "Home",
    "question_input": "",
    "clear_question_input": False,
    "learning_cache": {},
    "evaluation_results": {},
    "generated_evaluation_cases": {},
    "evaluation_errors": {},
    "preview_document_id": None,
    "preview_page": 1,
    "gemini_request_count": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = st.secrets.get("GEMINI_API_KEY", "")

MODEL_NAME = st.secrets.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)
# ============================================================
# BASIC API USAGE PROTECTION
# ============================================================

MAX_GEMINI_REQUESTS_PER_SESSION = 10


def allow_gemini_request():
    """Limit Gemini API requests within one visitor session."""

    current_count = st.session_state.gemini_request_count

    if current_count >= MAX_GEMINI_REQUESTS_PER_SESSION:
        st.warning(
            "You have reached the Gemini request limit for this session."
        )
        return False

    st.session_state.gemini_request_count += 1
    return True


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_document_id(file_bytes):
    """Generate a unique ID based on the uploaded PDF content."""
    return hashlib.md5(file_bytes).hexdigest()


def process_document(uploaded_file):
    """Extract, chunk and store an uploaded PDF."""

    file_bytes = uploaded_file.getvalue()
    doc_id = make_document_id(file_bytes)

    if doc_id in st.session_state.documents:
        return False, "This document is already in your library."

    with st.spinner(f"Processing {uploaded_file.name}..."):

        pages = extract_pdf_pages(file_bytes)

        chunks = split_into_chunks(
            pages,
            uploaded_file.name,
        )

        if not chunks:
            return False, "No usable text was found in this document."

        with fitz.open(stream=file_bytes, filetype="pdf") as pdf_document:
            total_page_count = len(pdf_document)

        st.session_state.documents[doc_id] = {
            "id": doc_id,
            "name": uploaded_file.name,
            "pages": pages,
            "pdf_bytes": file_bytes,
            "page_count": total_page_count,
            "uploaded_at": datetime.now().strftime(
                "%d %b %Y, %I:%M %p"
            ),
        }

        st.session_state.chunks[doc_id] = chunks

        # Invalidate cached indexes and overviews.
        st.session_state.retrieval_cache.clear()
        st.session_state.overviews.pop(doc_id, None)

        return True, f"{uploaded_file.name} added successfully."


def get_selected_documents():
    """Return selected document IDs that still exist."""

    return [
        doc_id
        for doc_id in st.session_state.selected_document_ids
        if doc_id in st.session_state.documents
    ]


def get_combined_retriever(selected_ids):
    """Build and cache a combined retriever for selected PDFs."""

    cache_key = tuple(sorted(selected_ids))

    if cache_key in st.session_state.retrieval_cache:
        return st.session_state.retrieval_cache[cache_key]

    combined_chunks = []

    for doc_id in selected_ids:
        for chunk in st.session_state.chunks.get(doc_id, []):
            enriched_chunk = dict(chunk)
            enriched_chunk["doc_id"] = doc_id
            combined_chunks.append(enriched_chunk)

    if not combined_chunks:
        return None, []

    retriever = build_retriever(combined_chunks)

    st.session_state.retrieval_cache[cache_key] = (
        retriever,
        combined_chunks,
    )

    return retriever, combined_chunks


def navigate_to(view):
    """Change navigation using a widget callback."""

    st.session_state.active_view = view


def open_document(doc_id):
    """Open a document in the Understand page."""

    st.session_state.selected_document_ids = [doc_id]
    st.session_state.active_view = "Understand"


def remove_document(doc_id):
    """Remove a document and invalidate dependent data."""

    st.session_state.documents.pop(doc_id, None)
    st.session_state.chunks.pop(doc_id, None)

    st.session_state.overviews.clear()
    st.session_state.retrieval_cache.clear()
    st.session_state.chat_histories.clear()
    st.session_state.learning_cache.clear()
    if st.session_state.preview_document_id == doc_id:
        st.session_state.preview_document_id = None
        st.session_state.preview_page = 1


# ============================================================
# TOP NAVIGATION (SIDEBAR-FREE)
# ============================================================

PAGES = [
    ("Home", "⌂ Home"),
    ("Understand", "✦ Understand"),
    ("Ask", "✎ Ask questions"),
    ("Learn", "▤ Learning"),
    ("Evaluation", "📊 Evaluation"),
    ("PDF Preview", "▧ PDF preview"),
    ("Library", "▣ Library"),
]


def render_top_navigation():
    st.markdown(
        '<div class="brand">🧠 KnowledgePilot</div>'
        '<div class="brand-caption">Your intelligent document workspace</div>',
        unsafe_allow_html=True,
    )
    nav_columns = st.columns([1, 1.2, 1.3, 1, 1.15, 1.2, 1])
    for column, (page, label) in zip(nav_columns, PAGES):
        with column:
            is_active = st.session_state.active_view == page
            st.button(
                label,
                key=f"topnav_{page}",
                on_click=navigate_to,
                args=(page,),
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
    if st.session_state.documents:
        valid_ids = [
            doc_id for doc_id in st.session_state.selected_document_ids
            if doc_id in st.session_state.documents
        ]
        st.session_state.selected_document_ids = valid_ids
        st.multiselect(
            "Documents used by Understand, Ask and Learning",
            options=list(st.session_state.documents.keys()),
            format_func=lambda doc_id: st.session_state.documents[doc_id]["name"],
            key="selected_document_ids",
            help="Choose one or more PDFs for AI features. You can still preview any PDF from PDF Preview.",
            placeholder="Choose the PDFs you want to work with",
        )
    st.divider()


render_top_navigation()


# ============================================================
# REUSABLE UI COMPONENTS
# ============================================================

def page_header(title, subtitle):

    st.markdown(
        f"""
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value):

    st.markdown(
        f"""
        <div class="custom-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def document_card(doc_id, document):
    safe_name = html.escape(str(document["name"]))
    safe_uploaded_at = html.escape(str(document["uploaded_at"]))

    st.markdown(
        f"""
        <div class="custom-card">
            <div class="document-title">
                📄 {safe_name}
            </div>
            <div class="muted-text">
                {document["page_count"]} pages
                &nbsp; • &nbsp;
                Added {safe_uploaded_at}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HOME PAGE
# ============================================================

def render_home():
    total_documents = len(st.session_state.documents)
    total_pages = sum(doc["page_count"] for doc in st.session_state.documents.values())

    st.markdown(
        """
        <div class="hero-panel">
            <div class="section-label">YOUR AI DOCUMENT WORKSPACE</div>
            <h1 style="font-size:2.35rem; margin:8px 0 10px 0;">Your PDFs. Clear answers.</h1>
            <p style="color:#CBD5E1; font-size:1.05rem; max-width:720px;">
                Upload a document to understand its key ideas, ask questions with source citations,
                create revision material, or inspect the original PDF pages.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 1. Upload your documents")
    st.caption("Upload one or more text-based PDFs. Your files are processed in this session.")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="home_pdf_uploader",
        help="Maximum file size: 25 MB per PDF.",
    )
    upload_col, upload_hint = st.columns([1, 2])
    with upload_col:
        process_clicked = st.button(
            "Upload and process PDFs →",
            type="primary",
            use_container_width=True,
            disabled=not bool(uploaded_files),
            key="home_process_pdfs",
        )
    with upload_hint:
        st.caption("KnowledgePilot extracts text and prepares it for search. AI-generated features run only when you request them.")

    if process_clicked and uploaded_files:

        oversized_files = [
            file.name
            for file in uploaded_files
            if file.size > 25 * 1024 * 1024
        ]

        if oversized_files:
            st.error(
                "These files exceed the 25 MB limit: "
                + ", ".join(oversized_files)
            )

        else:
            successes = 0

            for uploaded_file in uploaded_files:
                try:
                    success, message = process_document(uploaded_file)

                    if success:
                        successes += 1
                        st.success(message)

                        doc_id = make_document_id(
                            uploaded_file.getvalue()
                        )

                        if doc_id not in st.session_state.selected_document_ids:
                            st.session_state.selected_document_ids.append(doc_id)

                    else:
                        st.info(message)

                except Exception as error:
                    st.error(
                        f"Could not process {uploaded_file.name}: {error}"
                    )

            if successes:
                st.rerun()

    
    st.write("")
    metric_columns = st.columns(3)
    with metric_columns[0]:
        metric_card("Documents in this session", total_documents)
    with metric_columns[1]:
        metric_card("Pages processed", total_pages)
    with metric_columns[2]:
        metric_card("Selected for AI", len(get_selected_documents()))

    if st.session_state.documents:
        st.markdown("### Recently added")
        recent_documents = list(st.session_state.documents.items())[-4:]
        for doc_id, document in reversed(recent_documents):
            col1, col2 = st.columns([5, 1])
            with col1:
                document_card(doc_id, document)
            with col2:
                st.write("")
                st.button(
                    "Open",
                    key=f"home_open_{doc_id}",
                    on_click=open_document,
                    args=(doc_id,),
                    use_container_width=True,
                )
    else:
        st.info("Start by uploading a PDF above. Once it is processed, the document tools will be ready to use.")


# ============================================================
# UNDERSTAND PAGE
# ============================================================

def render_understand():

    page_header(
        "Understand your documents",
        "Generate structured insights from your PDF documents.",
    )

    selected_ids = get_selected_documents()

    if not selected_ids:

        st.info(
            "Choose at least one PDF in the document selector above, or upload one from Home."
        )
        return

    if not API_KEY:

        st.error(
            "Configure your Gemini API key before generating an overview."
        )
        return

    for doc_id in selected_ids:

        document = st.session_state.documents[doc_id]

        st.markdown(
            f"### 📄 {document['name']}"
        )

        st.caption(
            f"{document['page_count']} pages"
        )

        if doc_id not in st.session_state.overviews:

            if st.button(
                "Generate document overview",
                key=f"overview_{doc_id}",
                use_container_width=True,
            ):

                try:

                    with st.spinner(
                        "Analyzing document with Gemini..."
                    ):

                        overview = generate_document_overview(
                            document["pages"],
                            API_KEY,
                            MODEL_NAME,
                        )

                    st.session_state.overviews[doc_id] = overview
                    st.rerun()

                except Exception as error:

                    st.error(
                        classify_gemini_error(error)
                    )

        else:

            overview = st.session_state.overviews[doc_id]

            with st.container(border=True):
                st.markdown("#### Overview")
                st.write(
                    overview.get(
                        "overview",
                        "No overview available.",
                    )
                )

            st.markdown("#### Key topics")

            topics = overview.get("key_topics", [])

            if topics:
                for topic in topics:
                    st.markdown(f"- {topic}")
            else:
                st.caption("No key topics returned.")

            st.markdown("#### Key findings")

            findings = overview.get("key_findings", [])

            if findings:
                for finding in findings:
                    st.markdown(f"- {finding}")
            else:
                st.caption("No key findings returned.")

            st.markdown("#### Document structure")

            structure = overview.get(
                "document_structure",
                [],
            )

            if structure:
                for item in structure:
                    if isinstance(item, dict):
                        section = item.get("section", "Section")
                        description = item.get("description", "")
                        st.markdown(f"- **{section}:** {description}")
                    else:
                        st.markdown(f"- {item}")
            else:
                st.caption("No document structure returned.")

            st.markdown("#### Suggested questions")

            questions = overview.get(
                "suggested_questions",
                [],
            )

            for index, question in enumerate(questions):

                if st.button(
                    question,
                    key=f"suggested_{doc_id}_{index}",
                    use_container_width=True,
                ):

                    st.session_state.question_input = question
                    st.session_state.selected_document_ids = [doc_id]
                    st.session_state.active_view = "Ask"
                    st.rerun()

            if st.button(
                "Generate document overview",
                key=f"overview_{doc_id}",
                use_container_width=True,
            ):
                if not allow_gemini_request():
                    return

                try:
                    with st.spinner("Analyzing document with Gemini..."):
                        overview = generate_document_overview(
                            document["pages"],
                            API_KEY,
                            MODEL_NAME,
                        )

                    st.session_state.overviews[doc_id] = overview
                    st.rerun()

                except Exception as error:
                    st.error(classify_gemini_error(error))


# ============================================================
# ASK PAGE
# ============================================================

def render_ask():

    page_header(
        "Ask your documents",
        "Get answers based on the information in your PDFs.",
    )

    selected_ids = get_selected_documents()

    if not selected_ids:

        st.info(
            "Choose one or more PDFs in the document selector above, or upload one from Home."
        )
        return

    if not API_KEY:

        st.error(
            "Configure your Gemini API key before asking questions."
        )
        return

    history_key = tuple(sorted(selected_ids))

    if history_key not in st.session_state.chat_histories:
        st.session_state.chat_histories[history_key] = []

    history = st.session_state.chat_histories[history_key]

    # Display previous messages.
    for message_index, message in enumerate(history):

        with st.chat_message(message["role"]):
            st.markdown(message["content"])
                        # Display locally validated citation status.
            citation_result = message.get("citation_result")

            if citation_result:

                if citation_result["status"] == "valid":
                    st.success(citation_result["message"])

                elif citation_result["status"] == "missing":
                    st.warning(citation_result["message"])

                else:
                    st.error(citation_result["message"])

                if citation_result.get("invalid_citations"):
                    st.write(
                        "Invalid evidence IDs:",
                        citation_result["invalid_citations"]
                    )

                if citation_result.get("mismatched_citations"):
                    st.write(
                        "Citations with incorrect source/page details:",
                        citation_result["mismatched_citations"]
                    )

            # Optional answer evaluation: costs one Gemini request.
            if message.get("sources") and message.get("question"):

                if st.button(
                    "Evaluate answer quality",
                    key=f"evaluate_answer_{message_index}",
                    help="Evaluate faithfulness and answer relevance using one Gemini request."
                ):
                    if not allow_gemini_request():
                        return

                    try:
                        with st.spinner("Evaluating answer quality..."):

                            evaluation_chunks = [
                                {
                                    "source": item.get("filename", "Unknown"),
                                    "page": item.get("page", "?"),
                                    "text": item.get("text", ""),
                                }
                                for item in message["sources"]
                            ]

                            evaluation = evaluate_answer(
                                question=message["question"],
                                answer=message["content"],
                                retrieved_chunks=evaluation_chunks,
                                api_key=API_KEY,
                                model_name=MODEL_NAME,
                            )

                            message["answer_evaluation"] = evaluation

                        st.rerun()

                    except Exception as error:
                        st.error(classify_gemini_error(error))

            # Display saved answer evaluation.
            evaluation = message.get("answer_evaluation")

            if evaluation:

                st.markdown("#### Answer quality evaluation")

                score_col1, score_col2 = st.columns(2)

                with score_col1:
                    st.metric(
                        "Faithfulness",
                        f"{evaluation['faithfulness_score']}/5"
                    )

                with score_col2:
                    st.metric(
                        "Answer relevance",
                        f"{evaluation['relevance_score']}/5"
                    )

                st.markdown("**Faithfulness explanation**")
                st.write(evaluation["faithfulness_explanation"])

                st.markdown("**Answer relevance explanation**")
                st.write(evaluation["relevance_explanation"])

                if evaluation["unsupported_claims"]:
                    st.markdown("**Unsupported claims identified**")
                    for claim in evaluation["unsupported_claims"]:
                        st.write(f"- {claim}")
                else:
                    st.success("No unsupported claims were identified by the evaluator.")

            if message.get("sources"):

                with st.expander("View source passages"):

                    for source_index, source in enumerate(message["sources"]):

                        st.markdown(
                            f"**{source.get('filename', 'Document')} — Page {source.get('page', '?')}**"
                        )
                        if source.get("score") is not None:
                            st.caption(
                                f"Hybrid relevance: {source['score']:.3f} · "
                                f"Semantic: {source.get('semantic_score', 0):.3f} · "
                                f"Keyword: {source.get('lexical_score', 0):.3f}"
                            )
                        st.write(source.get("text", ""))
                        if source.get("doc_id") and st.button(
                            "Open cited page",
                            key=f"history_preview_{message_index}_{source_index}_{source.get('doc_id')}_{source.get('page')}",
                        ):
                            st.session_state.preview_document_id = source["doc_id"]
                            st.session_state.preview_page = int(source.get("page", 1) or 1)
                            st.session_state[f"preview_page_input_{source['doc_id']}"] = int(source.get("page", 1) or 1)
                            st.session_state.active_view = "PDF Preview"
                            st.rerun()

    st.markdown("### Ask a question")
    st.caption("Answers are based on the selected PDFs. Expand source passages to inspect evidence or open the cited page.")

    # Clear the input before rendering the widget.
    if st.session_state.clear_question_input:

        st.session_state.question_input = ""
        st.session_state.clear_question_input = False

    question = st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="Ask something about your selected documents...",
        label_visibility="collapsed",
    )

    if st.button(
        "Send question →",
        use_container_width=True,
    ):
        question = question.strip()

        if not question:
            st.warning("Please enter a question.")

        elif not allow_gemini_request():
            pass

        else:
            try:
                with st.spinner(
                    "Searching documents and generating your answer..."
                ):
                    retriever, combined_chunks = get_combined_retriever(
                        selected_ids
                    )

                    if retriever is None:
                        st.error("No document content is available.")
                        return

                    results = retrieve_chunks(
                        question,
                        combined_chunks,
                        retriever,
                        top_k=5,
                    )

                    answer = generate_with_gemini(
                        question,
                        results,
                        API_KEY,
                        MODEL_NAME,
                    )

                    citation_result = validate_citations(
                        answer,
                        results,
                    )

                sources = []

                for result in results:
                    sources.append({
                        "filename": result.get(
                            "source",
                            result.get("filename", "Document")
                        ),
                        "doc_id": result.get("doc_id"),
                        "page": result.get("page", "?"),
                        "text": result.get("text", ""),
                        "score": result.get("score"),
                        "semantic_score": result.get("semantic_score"),
                        "lexical_score": result.get("lexical_score"),
                    })

                history.append({
                    "role": "user",
                    "content": question,
                })

                history.append({
                    "role": "assistant",
                    "content": answer,
                    "question": question,
                    "sources": sources,
                    "citation_result": citation_result,
                    "answer_evaluation": None,
                })

                st.session_state.clear_question_input = True
                st.rerun()

            except Exception as error:
                st.error(classify_gemini_error(error))




# ============================================================
# LEARNING WORKSPACE
# ============================================================
def render_learn():
    page_header(
        "Learning workspace",
        "Turn your selected PDFs into explanations, revision notes, and practice questions.",
    )
    selected_ids = get_selected_documents()
    if not selected_ids:
        st.info("Choose one or more PDFs in the document selector above, or upload one from Home.")
        return
    if not API_KEY:
        st.error("Configure your Gemini API key before generating learning material.")
        return

    mode = st.selectbox("Learning activity", ["Explanation", "Revision notes", "Quiz"])
    focus = st.text_input("Topic or focus (optional)", placeholder="e.g. neural networks, chapter 2, key definitions")
    st.caption("Generation happens only when you click the button. Results are cached for this session.")
    if st.button(
        "Generate learning material",
        type="primary",
        use_container_width=True,
    ):
        if not allow_gemini_request():
            pass
        else:
            query = focus.strip() or {
                "Explanation": "Explain the main concepts in these documents",
                "Revision notes": "Summarize the key concepts and definitions",
                "Quiz": "Identify the key concepts suitable for a practice quiz",
            }[mode]

            try:
                with st.spinner("Finding relevant passages..."):
                    retriever, combined_chunks = get_combined_retriever(
                        selected_ids
                    )

                    if retriever is None:
                        st.warning("No readable document content is available.")
                        return

                    results = retrieve_chunks(
                        query,
                        combined_chunks,
                        retriever,
                        top_k=6,
                    )

                    output = generate_learning_material(
                        mode,
                        focus,
                        results,
                        API_KEY,
                        MODEL_NAME,
                    )

                cache_key = (
                    tuple(sorted(selected_ids)),
                    mode,
                    focus.strip().lower(),
                )

                st.session_state.learning_cache[cache_key] = {
                    "output": output,
                    "sources": results,
                }

            except Exception as error:
                st.error(classify_gemini_error(error))
    cache_key = (tuple(sorted(selected_ids)), mode, focus.strip().lower())
    cached = st.session_state.learning_cache.get(cache_key)
    if cached:
        st.markdown("### Your study material")
        with st.container(border=True):
            st.markdown(cached["output"])
        with st.expander("Evidence used · source passages"):
            for item in cached.get("sources", []):
                st.markdown(f"**{item.get('source', 'Document')} — Page {item.get('page', '?')}**")
                st.write(item.get("text", ""))
# ============================================================
# RETRIEVAL EVALUATION
# ============================================================
def build_local_evaluation_cases(chunks, questions_per_document=3):
    """
    Build a local retrieval smoke-test dataset without using an LLM.

    Each test case uses a short excerpt from an existing chunk as the
    search query and labels that chunk's source/page as relevant.
    These cases test retrieval mechanics, not semantic understanding.
    """
    cases = []
    seen = set()
    counts = {}

    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page")
        text = chunk.get("text", "").strip()

        if not text or page is None:
            continue

        source_key = str(source).casefold()
        if counts.get(source_key, 0) >= questions_per_document:
            continue

        # Use a short excerpt as a deterministic retrieval query.
        words = text.split()
        if len(words) < 5:
            continue

        query_text = " ".join(words[:12]).strip()
        dedupe_key = (source_key, page, query_text.casefold())

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        cases.append({
            "question": (
                "Find the document passage containing this excerpt: "
                f"{query_text}"
            ),
            "expected_answer": (
                "The relevant passage should contain the supplied excerpt."
            ),
            "relevant_passages": [
                {
                    "source": source,
                    "page": page,
                }
            ],
        })

        counts[source_key] = counts.get(source_key, 0) + 1

    return cases


def render_evaluation():
    st.header("Retrieval Evaluation")

    st.write(
        "Measure how effectively KnowledgePilot retrieves "
        "relevant document passages."
    )

    selected_ids = st.session_state.selected_document_ids

    if not selected_ids:
        st.info("Upload and select at least one PDF to begin evaluation.")
        return

    # Get chunks and retriever for selected documents.
    retriever, chunks = get_combined_retriever(selected_ids)
    
    if not chunks or retriever is None:
        st.warning("No document chunks are available for evaluation.")
        return

    # Build a stable cache key for the selected document set.
    cache_key = tuple(sorted(selected_ids))

    if "generated_evaluation_cases" not in st.session_state:
        st.session_state.generated_evaluation_cases = {}

    if "evaluation_errors" not in st.session_state:
        st.session_state.evaluation_errors = {}

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}

    if st.session_state.get("evaluation_dataset_type") == "local_smoke_test":
        st.caption(
            "Local retrieval smoke-test dataset. Queries are derived from "
            "document excerpts; results do not measure natural-language "
            "question answering quality."
    )
    else:
        st.caption(
            "Evaluation questions are generated from your selected PDFs. "
            "Their reference labels are AI-generated and may need review."
        )

    # Generate questions only once for this document selection.
    if cache_key not in st.session_state.generated_evaluation_cases:

        st.info(
            "Generate an evaluation dataset based on your selected PDFs."
        )

        if st.button(
            "Generate Evaluation Questions",
            key="generate_evaluation_questions",
        ):
            if not allow_gemini_request():
                return
            api_key = st.secrets.get("GEMINI_API_KEY", "")
            model_name = st.secrets.get(
                "GEMINI_MODEL",
                "gemini-3.5-flash",
            )

            if not api_key:
                st.error("Gemini API key is not configured.")
                return

            
            try:
                with st.spinner(
                    "Generating document-specific evaluation questions..."
                ):
                    cases = generate_evaluation_cases(
                        chunks=chunks,
                        api_key=api_key,
                        model_name=model_name,
                        questions_per_document=3,
                    )

                if not cases:
                    raise ValueError(
                        "Gemini returned no valid evaluation questions."
                    )

                st.session_state.generated_evaluation_cases[
                    cache_key
                ] = cases

                st.session_state.evaluation_errors.pop(
                    cache_key,
                    None,
                )

                st.session_state.evaluation_results.pop(
                    cache_key,
                    None,
                )

                st.rerun()

            except Exception as exc:
                error_message = str(exc)
                st.warning(
                    "Gemini could not generate evaluation questions. "
                    "Trying a local retrieval smoke-test dataset instead."
                )

                try:
                    local_cases = build_local_evaluation_cases(
                        chunks,
                        questions_per_document=3,
                    )

                    if not local_cases:
                        raise ValueError(
                            "No suitable text chunks were found."
                        )

                    st.session_state.generated_evaluation_cases[
                        cache_key
                    ] = local_cases

                    st.session_state.evaluation_errors.pop(
                        cache_key,
                        None,
                    )

                    st.session_state.evaluation_results.pop(
                        cache_key,
                        None,
                    )

                    st.session_state.evaluation_dataset_type = (
                        "local_smoke_test"
                    )

                    st.session_state.evaluation_fallback_message = (
                        error_message
                    )

                    st.rerun()

                except Exception as fallback_exc:
                    st.session_state.evaluation_errors[cache_key] = (
                        f"Gemini error: {error_message}; "
                        f"Local fallback error: {fallback_exc}"
                    )


        error = st.session_state.evaluation_errors.get(cache_key)

        if error:
            st.error(f"Could not generate evaluation questions: {error}")

        return

    # Retrieve the generated evaluation dataset.
    test_cases = st.session_state.generated_evaluation_cases[cache_key]
    if st.session_state.get("evaluation_dataset_type") == "local_smoke_test":
        st.info(
            "Using local fallback cases because Gemini question generation "
            "was unavailable. These tests require no Gemini API calls."
        )

    st.subheader("Generated Evaluation Dataset")

    st.write(f"Total evaluation questions: **{len(test_cases)}**")

    for index, case in enumerate(test_cases, start=1):
        with st.expander(
            f"Question {index}: {case['question']}"
        ):
            st.write("**Expected answer:**")
            st.write(case.get("expected_answer", "Not available"))

            st.write("**Reference passages:**")

            for passage in case["relevant_passages"]:
                st.write(
                    f"- {passage['source']} — Page {passage['page']}"
                )

    st.warning(
        "These questions and reference passages were generated by AI. "
        "Review them before interpreting the evaluation scores."
    )

    col1, col2 = st.columns(2)

    with col1:
        run_evaluation = st.button(
            "Run Retrieval Evaluation",
            key="run_retrieval_evaluation",
        )

    with col2:
        regenerate = st.button(
            "Regenerate Questions",
            key="regenerate_evaluation_questions",
        )

    if regenerate:
        st.session_state.generated_evaluation_cases.pop(
            cache_key,
            None,
        )
        st.session_state.evaluation_results.pop(
            cache_key,
            None,
        )
        st.session_state.evaluation_errors.pop(
            cache_key,
            None,
        )
        st.rerun()

    if run_evaluation:
        try:
            with st.spinner("Evaluating retrieval strategies..."):
                results = evaluate_retrieval_strategies(
                    test_cases=test_cases,
                    chunks=chunks,
                    retriever=retriever,
                    top_k=5,
                )

            st.session_state.evaluation_results[cache_key] = results

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

    # Display saved results.
    results = st.session_state.evaluation_results.get(cache_key)

    if not results:
        return

    st.divider()
    st.subheader("Retrieval Strategy Comparison")

    comparison_data = []

    for strategy, metrics in results.items():
        comparison_data.append(
            {
                "Strategy": strategy,
                "Hit Rate@5": round(metrics["hit_rate_at_k"], 3),
                "Recall@5": round(metrics["recall_at_k"], 3),
                "MRR": round(metrics["mrr"], 3),
            }
        )

    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Scores range from 0 to 1. Higher scores indicate better "
        "retrieval performance on this generated evaluation dataset."
    )

    st.subheader("Individual Question Results")

    for strategy, metrics in results.items():
        with st.expander(strategy):
            for index, result in enumerate(
                metrics["individual_results"],
                start=1,
            ):
                st.markdown(
                    f"**Question {index}:** {result['question']}"
                )

                st.write(
                    f"Hit Rate@5: {result['hit_rate']:.2f}"
                )

                st.write(
                    f"Recall@5: {result['recall']:.2f}"
                )

                st.write(
                    f"Reciprocal Rank: "
                    f"{result['reciprocal_rank']:.2f}"
                )

                st.write("Retrieved passages:")

                for passage in result["retrieved_passages"]:
                    st.write(
                        f"- {passage['source']} — "
                        f"Page {passage['page']}"
                    )

                st.divider()



# ============================================================
# PDF PAGE PREVIEW
# ============================================================
def render_pdf_preview():
    page_header("PDF page preview", "Inspect the original page behind a citation.")
    documents = st.session_state.documents
    if not documents:
        st.info("Upload a PDF to preview its pages.")
        return

    doc_ids = list(documents.keys())
    default_id = st.session_state.preview_document_id
    if default_id not in documents:
        selected = get_selected_documents()
        default_id = selected[0] if selected else doc_ids[0]
    selected_id = st.selectbox(
        "Document", doc_ids,
        index=doc_ids.index(default_id),
        format_func=lambda doc_id: documents[doc_id]["name"],
        key="preview_select_document",
    )
    document = documents[selected_id]
    requested_page = st.session_state.preview_page if selected_id == st.session_state.preview_document_id else 1
    page_number = st.number_input(
        "Page number", min_value=1, max_value=max(1, document["page_count"]),
        value=min(max(1, int(requested_page or 1)), document["page_count"]),
        step=1,
        key=f"preview_page_input_{selected_id}",
    )
    st.session_state.preview_document_id = selected_id
    st.session_state.preview_page = int(page_number)
    pdf_bytes = document.get("pdf_bytes")
    if not pdf_bytes:
        st.warning("The PDF bytes are not available for this session. Re-upload the document to enable preview.")
        return
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            page = pdf.load_page(int(page_number) - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            png_bytes = pixmap.tobytes("png")
        st.image(png_bytes, caption=f"{document['name']} · Page {int(page_number)}", use_container_width=True)
        st.download_button(
            "Download original PDF", data=pdf_bytes, file_name=document["name"],
            mime="application/pdf", use_container_width=True,
        )
    except Exception as error:
        st.error(f"Could not render this PDF page: {error}")


# ============================================================
# LIBRARY PAGE
# ============================================================

def render_library():

    page_header(
        "Document library",
        "Manage and explore all your uploaded documents.",
    )

    if not st.session_state.documents:

        st.info(
            "No documents available yet. Go to Home and upload a PDF to get started."
        )
        return

    search_query = st.text_input(
        "Search documents",
        placeholder="Search by filename...",
    )

    filtered_documents = [
        (doc_id, document)
        for doc_id, document in st.session_state.documents.items()
        if search_query.lower() in document["name"].lower()
    ]

    st.caption(
        f"{len(filtered_documents)} document(s) found"
    )

    if not filtered_documents:

        st.info("No matching documents found.")
        return

    for doc_id, document in filtered_documents:

        document_card(doc_id, document)

        col1, col2 = st.columns([1, 1])

        with col1:

            st.button(
                "Understand",
                key=f"library_understand_{doc_id}",
                on_click=open_document,
                args=(doc_id,),
                use_container_width=True,
            )

        with col2:

            st.button(
                "Remove",
                key=f"library_remove_{doc_id}",
                on_click=remove_document,
                args=(doc_id,),
                use_container_width=True,
            )

        st.write("")


# ============================================================
# PAGE ROUTING
# ============================================================

active_view = st.session_state.active_view

if active_view == "Home":

    render_home()

elif active_view == "Understand":

    render_understand()

elif active_view == "Ask":

    render_ask()

elif active_view == "Learn":

    render_learn()
elif active_view == "Evaluation":

    render_evaluation()

elif active_view == "PDF Preview":

    render_pdf_preview()

elif active_view == "Library":

    render_library()
