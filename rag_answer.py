import sys
import ollama
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
# ==========================================

_INDEX = None



def score_retrieval_confidence(nodes):
    if not nodes:
        return 0.0

    scores = []
    files = set()

    for n in nodes:
        if hasattr(n, "score") and n.score is not None:
            scores.append(n.score)
        meta = n.node.metadata
        files.add(meta.get("file_path", "unknown"))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    file_focus = 1 / len(files) if files else 0.0

    # Weighted heuristic
    confidence = (avg_score * 0.7) + (file_focus * 0.3)
    return round(confidence, 2)


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

    scores = {}
    for intent, keywords in signals.items():
        scores[intent] = sum(1 for k in keywords if k in q)

    total_hits = sum(scores.values())
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if total_hits == 0:
        return {
            "intent": "ambiguous",
            "confidence": 0.0,
            "reason": "No intent keywords matched"
        }

    confidence = best_score / total_hits

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


def build_clarifying_prompt(query, nodes):
    return f"""
        You are unsure you fully understand the user's intent or lack sufficient evidence.

        DO NOT answer the question yet.

        Instead:
        - Identify what is ambiguous
        - Ask 1–2 clarifying questions
        - Explain briefly why clarification is needed

        Original question:
        {query}

        Relevant code snippets:
        {format_context(nodes)}

        Respond ONLY with clarifying questions.
        """

def add_uncertainty_header(confidence):
    if confidence > 0.7:
        return ""
    return (
        "⚠️ **Confidence Notice**:\n"
        "This answer is based on partial evidence from the codebase. "
        "Some assumptions may be incomplete.\n\n"
    )


def should_ask_clarifying_question(
    intent_confidence: float,
    retrieval_confidence: float
):
    if intent_confidence < 0.5:
        return True
    if retrieval_confidence < 0.35:
        return True
    return False


# ---------- Index + Retrieval ----------

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



# ---------- Prompt Helpers ----------

def format_context(nodes):
    blocks = []
    for node in nodes:
        meta = node.node.metadata
        path = meta.get("file_path", "unknown")
        text = node.node.text.strip()
        blocks.append(f"FILE: {path}\n{text}")
    return "\n\n".join(blocks)


def build_architecture_prompt(query, nodes):
    return f"""
You are a senior software architect onboarding to an unfamiliar codebase.

Your goal is to explain system behavior and architecture.

When answering:
1. Identify the entry point (request, event, user action)
2. Trace control flow across layers
3. Explain responsibility boundaries
4. Describe backend ↔ frontend interaction
5. Mention key files only when they clarify the flow

Do NOT:
- Restate code line-by-line
- Speculate beyond the provided code

QUESTION:
{query}

CODE CONTEXT:
{format_context(nodes)}

Provide a clear, end-to-end explanation.
"""


def build_bug_prompt(query, nodes):
    return f"""
You are a senior engineer debugging a production issue.

Your task:
- Identify the most likely root cause
- Explain why the behavior occurs
- Suggest a fix grounded in the code

When answering:
1. Identify where the failure originates
2. Trace how the bug propagates
3. Explain incorrect assumptions
4. Propose a concrete fix

Do NOT:
- Guess without evidence
- Suggest unrelated changes

QUESTION:
{query}

CODE CONTEXT:
{format_context(nodes)}

Analyze the bug step-by-step.
"""


def build_refactor_prompt(query, nodes):
    return f"""
You are a senior engineer reviewing code quality and design.

Your task:
- Identify structural improvements
- Explain why the current design is suboptimal
- Suggest refactors that improve clarity and maintainability

When answering:
1. Identify problematic patterns
2. Explain risks or limitations
3. Propose specific refactors
4. Explain tradeoffs

Do NOT:
- Rewrite everything
- Focus on micro-optimizations

QUESTION:
{query}

CODE CONTEXT:
{format_context(nodes)}

Provide actionable refactoring advice.
"""


def build_general_prompt(query, nodes):
    return f"""
You are a senior software engineer answering a question about a codebase.

QUESTION:
{query}

CODE CONTEXT:
{format_context(nodes)}

Answer clearly and concisely.
"""


# ---------- Main ----------

def main(query: str):
    print("🔍 Retrieving relevant code...")
    nodes = retrieve_context(query)


    intent_info = detect_question_type(query)
    intent = intent_info["intent"]
    intent_conf = intent_info["confidence"]
    retrieval_conf = score_retrieval_confidence(nodes)

    print(f"🧠 Intent: {intent} (confidence {intent_conf})")
    print(f"📚 Evidence confidence: {retrieval_conf}")

    if should_ask_clarifying_question(intent_conf, retrieval_conf):
        prompt = build_clarifying_prompt(query, nodes)
    else:
        if intent == "architecture":
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

    answer = response["message"]["content"]
    header = add_uncertainty_header(retrieval_conf)

    print(header + answer)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rag_answer.py \"your question here\"")
        sys.exit(1)

    main(sys.argv[1])
