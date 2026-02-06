import sys
import ollama
from pathlib import Path
import yaml

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ================= CONFIG =================
INDEX_DIR = "/home/achilles/Dev/index"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"
OLLAMA_MODEL = "qwen2.5-coder:7b"
TOP_K = 6
ASSUMPTIONS_DIR = Path("assumptions")
# ==========================================

_INDEX = None


# ---------- Index Handling ----------

def get_index():
    global _INDEX
    if _INDEX is None:
        storage_context = StorageContext.from_defaults(
            persist_dir=INDEX_DIR
        )
        _INDEX = load_index_from_storage(storage_context)
    return _INDEX


def retrieve_context(query: str):
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL,
        trust_remote_code=True,
    )
    Settings.llm = None  # 🔒 disable OpenAI fallback

    index = get_index()
    query_engine = index.as_query_engine(
        similarity_top_k=TOP_K,
        llm=None,
    )

    response = query_engine.query(query)
    return response.source_nodes


# ---------- Intent Detection ----------

def detect_question_type(query: str):
    q = query.lower()

    signals = {
        "architecture": [
            "how does", "flow", "architecture", "end to end",
            "interaction", "lifecycle", "from backend", "to frontend"
        ],
        "bug": [
            "bug", "error", "exception", "fails",
            "not working", "traceback", "why is"
        ],
        "refactor": [
            "refactor", "clean up", "improve",
            "optimize", "rewrite", "better way"
        ],
    }

    scores = {k: 0 for k in signals}

    for intent, keywords in signals.items():
        scores[intent] = sum(1 for k in keywords if k in q)

    total = sum(scores.values())

    if total == 0:
        return {
            "intent": "ambiguous",
            "confidence": 0.0,
            "reason": "No intent keywords matched"
        }

    best_intent = max(scores, key=scores.get)
    confidence = scores[best_intent] / total

    if confidence < 0.5:
        return {
            "intent": "ambiguous",
            "confidence": round(confidence, 2),
            "reason": "Multiple intents detected"
        }

    return {
        "intent": best_intent,
        "confidence": round(confidence, 2),
        "reason": None
    }


# ---------- Retrieval Confidence ----------

def score_retrieval_confidence(nodes):
    if not nodes:
        return 0.0

    scores = []
    files = set()

    for n in nodes:
        if hasattr(n, "score") and n.score is not None:
            scores.append(n.score)
        files.add(n.node.metadata.get("file_path", "unknown"))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    file_focus = 1 / len(files) if files else 0.0

    return round((avg_score * 0.7) + (file_focus * 0.3), 2)


# ---------- Assumptions (Config-driven) ----------

def load_assumptions(intent: str):
    path = ASSUMPTIONS_DIR / f"{intent}.yaml"
    if not path.exists():
        return []

    with open(path) as f:
        data = yaml.safe_load(f)

    return data.get("assumptions", [])


def evaluate_assumptions(nodes, assumptions):
    evidence_text = " ".join(n.node.text.lower() for n in nodes)
    failed = []

    for a in assumptions:
        if not any(h in evidence_text for h in a["evidence_hints"]):
            failed.append(a["description"])

    return failed


def adjust_confidence(base, failed):
    penalty = 0.15 * len(failed)
    return max(0.0, round(base - penalty, 2))


# ---------- Prompt Builders ----------

def format_context(nodes):
    blocks = []
    for n in nodes:
        path = n.node.metadata.get("file_path", "unknown")
        blocks.append(f"FILE: {path}\n{n.node.text.strip()}")
    return "\n\n".join(blocks)


def build_architecture_prompt(query, nodes):
    return f"""
        You are a senior software architect.

        Explain system behavior using ONLY the provided code.

        Rules:
        - Trace execution flow
        - Explain boundaries and responsibilities
        - Do NOT assume missing components

        QUESTION:
        {query}

        CODE:
        {format_context(nodes)}
        """


def build_bug_prompt(query, nodes):
    return f"""
        You are debugging a production issue.

        Identify root cause using ONLY the provided code.

        QUESTION:
        {query}

        CODE:
        {format_context(nodes)}
        """


def build_refactor_prompt(query, nodes):
    return f"""
        You are reviewing code quality.

        Propose refactors justified by the provided code.

        QUESTION:
        {query}

        CODE:
        {format_context(nodes)}
        """


def build_general_prompt(query, nodes):
    return f"""
        Answer the question using ONLY the code below.

        QUESTION:
        {query}

        CODE:
        {format_context(nodes)}
        """


def build_clarifying_prompt(query, nodes):
    return f"""
        The intent or evidence is unclear.

        Ask 1–2 clarifying questions instead of answering.

        QUESTION:
        {query}

        CODE:
        {format_context(nodes)}
        """


# ---------- Notices ----------

def confidence_notice(confidence):
    if confidence >= 0.75:
        return ""

    if confidence >= 0.5:
        return (
            "⚠️ **Confidence Notice**:\n"
            "The retrieved code covers some relevant components, "
            "but does not show a complete end-to-end flow.\n\n"
        )

    if confidence >= 0.35:
        return (
            "⚠️ **Low Confidence**:\n"
            "Relevant code exists, but evidence is fragmented or incomplete.\n\n"
        )

    return (
        "❌ **Insufficient Evidence**:\n"
        "The codebase does not support a reliable answer.\n\n"
    )


def assumption_notice(failed):
    if not failed:
        return ""

    items = "\n".join(f"- {f}" for f in failed)
    return (
        "⚠️ **Unverified Assumptions**:\n"
        "The following claims are not directly supported by the code:\n"
        f"{items}\n\n"
    )


# ---------- Main ----------

def main(query: str):
    print("🔍 Retrieving relevant code...")
    nodes = retrieve_context(query)

    intent_info = detect_question_type(query)

    # Defensive normalization
    if isinstance(intent_info, str):
        intent_info = {
            "intent": intent_info,
            "confidence": 1.0,
            "reason": "legacy return"
        }

    intent = intent_info["intent"]
    intent_conf = intent_info["confidence"]

    retrieval_conf = score_retrieval_confidence(nodes)

    assumptions = load_assumptions(intent)
    failed_assumptions = evaluate_assumptions(nodes, assumptions)

    final_conf = adjust_confidence(retrieval_conf, failed_assumptions)

    print(f"🧠 Intent: {intent} (confidence {intent_conf})")
    print(f"📚 Evidence confidence: {final_conf}")

    # Hard refusal
    if final_conf < 0.35:
        print(confidence_notice(final_conf))
        print("❓ Not enough evidence. Please clarify or point to relevant files.")
        return

    # Clarify if intent ambiguous
    if intent == "ambiguous":
        prompt = build_clarifying_prompt(query, nodes)
    elif intent == "architecture":
        prompt = build_architecture_prompt(query, nodes)
    elif intent == "bug":
        prompt = build_bug_prompt(query, nodes)
    elif intent == "refactor":
        prompt = build_refactor_prompt(query, nodes)
    else:
        prompt = build_general_prompt(query, nodes)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    print()
    print(confidence_notice(final_conf) + assumption_notice(failed_assumptions))
    print(response["message"]["content"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_answer.py \"your question\"")
        sys.exit(1)

    main(sys.argv[1])
