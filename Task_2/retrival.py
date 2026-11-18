import os
import re
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# Optional: Yahoo Finance is only used for live stock-price queries.
try:
    import yfinance as yf
except ImportError:
    yf = None

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
INDEX_FOLDER = BASE_DIR / "rag_index"
EMBEDDINGS_PATH = INDEX_FOLDER / "embeddings.npy"
CHUNKS_PATH = INDEX_FOLDER / "chunks.json"
CONFIG_PATH = INDEX_FOLDER / "config.json"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_ID = "openai/gpt-oss-120b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY was not found. Add it to your .env file.")
if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
    raise FileNotFoundError(
        "RAG index files not found. Run indexing.py first to create rag_index/embeddings.npy and chunks.json."
    )

# Load the locally saved NumPy vector index and chunk metadata.
embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if len(chunks) != embeddings.shape[0]:
    raise ValueError(
        f"Index mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings. Re-run indexing.py."
    )

# Embeddings are normalized during indexing, so dot product = cosine similarity.
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = Groq(api_key=GROQ_API_KEY)

print(f"Loaded RAG index: {len(chunks)} chunks, {embeddings.shape[1]} dimensions")

# ----------------------------------------------------
# GREETINGS
# ----------------------------------------------------
GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]


def is_greeting(query: str) -> bool:
    q = query.lower().strip()
    return q in GREETINGS or any(q.startswith(g + " ") for g in GREETINGS)


# ----------------------------------------------------
# REALTIME STOCK FETCH
# ----------------------------------------------------
def fetch_realtime_stock(symbol: str):
    if yf is None:
        print("yfinance is not installed; live price lookup is unavailable.")
        return None
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        if data.empty:
            return None
        latest = data.iloc[-1]
        return {
            "symbol": symbol.upper(),
            "price": float(round(latest["Close"], 2)),
            "open": float(round(latest["Open"], 2)),
            "high": float(round(latest["High"], 2)),
            "low": float(round(latest["Low"], 2)),
            "currency": "INR" if symbol.endswith((".NS", ".BO")) else "USD",
        }
    except Exception as exc:
        print("REALTIME ERROR:", exc)
        return None


# ----------------------------------------------------
# SYMBOL EXTRACTION
# ----------------------------------------------------
YAHOO_INDIAN_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "SBI": "SBIN.NS",
    "WIPRO": "WIPRO.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
}


def extract_symbol(query: str):
    q = query.upper().replace(" ", "")
    for name, ticker in YAHOO_INDIAN_MAP.items():
        if name in q:
            return ticker
    words = query.replace("?", "").replace(",", "").split()
    for word in words:
        if word.upper().endswith((".NS", ".BO")):
            return word.upper()
    for word in words:
        cleaned = re.sub(r"[^A-Za-z]", "", word)
        if cleaned.isalpha() and 3 <= len(cleaned) <= 10:
            return f"{cleaned.upper()}.NS"
    return None


# ----------------------------------------------------
# INTENT DETECTION
# ----------------------------------------------------
def is_investment_advice(query: str) -> bool:
    q = query.lower()
    patterns = [
        r"should i invest", r"should i buy", r"is it.*good.*invest",
        r"is .* worth investing", r"is .* good buy", r"right time to buy",
        r"long term view", r"future prospects", r"investment analysis",
        r"investment view", r"fundamental analysis", r"is .* expected to grow",
        r"your view on",
    ]
    return any(re.search(pattern, q) for pattern in patterns)


def ask_realtime(query: str) -> bool:
    q = query.lower()
    patterns = [
        "price today", "live price", "current price", "market price", "price now",
        "latest price", "share price", "stock price", "price of", "price for",
        "realtime price",
    ]
    return any(pattern in q for pattern in patterns)


# ----------------------------------------------------
# NUMPY RAG RETRIEVAL (NO LANGCHAIN / QDRANT)
# ----------------------------------------------------
def retrieve_documents(query: str, top_k: int = TOP_K):
    """Return the top-k chunks ranked by cosine similarity to the query."""
    query_vector = embedding_model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)[0]

    # Index vectors and query vector are normalized; dot product is cosine similarity.
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]

    results = []
    for index in top_indices:
        item = dict(chunks[int(index)])
        item["score"] = float(scores[index])
        results.append(item)
    return results


def build_context(retrieved_chunks):
    context_parts = []
    seen = set()
    for rank, item in enumerate(retrieved_chunks, start=1):
        content = re.sub(r"\s+", " ", item.get("text", "")).strip()
        if not content or content in seen:
            continue
        seen.add(content)
        source = item.get("source", "Unknown source")
        page = item.get("page", "Unknown page")
        score = item.get("score", 0.0)
        context_parts.append(
            f"[Source {rank}: {source}, page {page}, similarity {score:.4f}]\n{content}"
        )
    return "\n\n".join(context_parts)


def rewrite_query(query: str) -> str:
    """Optional query rewriting; fall back to the original query if it fails."""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Rewrite the user's question as a concise search query. Preserve its meaning and important entities. Return only the rewritten query."},
                {"role": "user", "content": query},
            ],
            model=MODEL_ID,
            max_tokens=60,
            temperature=0.1,
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten or query
    except Exception as exc:
        print("Query rewrite failed; using original query:", exc)
        return query


def generate_rag_answer(user_query: str, investment_mode: bool = False, live_block: str = ""):
    rewritten = rewrite_query(user_query)
    retrieved = retrieve_documents(rewritten, top_k=TOP_K)
    context = build_context(retrieved)

    if not context:
        return "I could not find relevant information in the indexed documents. Please try rephrasing your question."

    if investment_mode:
        system_prompt = (
            "You are a careful financial information assistant. Use the supplied retrieved context "
            "and market snapshot where relevant. Do not invent facts. Distinguish source-supported "
            "information from general analysis. Do not give personalized buy/sell instructions."
        )
        user_prompt = f"""Retrieved context (cite the source filename and page number when using it):
{context}

{live_block}

Answer using these sections where relevant:
1. Executive Summary
2. Market Snapshot (only if data is supplied)
3. Fundamental Analysis
4. Valuation Overview (state when the retrieved context is insufficient)
5. Key Risk Factors
6. Neutral Outlook

User Query: {user_query}
"""
        max_tokens = 900
    else:
        system_prompt = (
            "You are a finance question-answering assistant. Answer using the retrieved context. "
            "Cite supporting source filenames and page numbers. If the context does not contain "
            "the answer, say that the indexed documents do not provide enough information. "
            "Do not fabricate citations or facts."
        )
        user_prompt = f"""Retrieved context:
{context}

Question: {user_query}

Give a clear answer and cite the relevant source filename and page number(s).
"""
        max_tokens = 700

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=MODEL_ID,
        max_tokens=max_tokens,
        temperature=0.15,
    )
    return response.choices[0].message.content


# ----------------------------------------------------
# MAIN RAG + FINANCE FUNCTION
# ----------------------------------------------------
def new_rag_ans(user_query: str):
    print("\n===== DEBUG START =====")
    print("RAW:", user_query)

    if is_greeting(user_query):
        print("===== DEBUG END =====")
        return "Hello! How can I help you with finance today?"

    # Investment questions can include a live market snapshot plus document retrieval.
    if is_investment_advice(user_query):
        print("DEBUG: Investment query detected")
        symbol = extract_symbol(user_query)
        live = fetch_realtime_stock(symbol) if symbol else None
        live_block = ""
        if live:
            live_block = f"""Real-Time Market Snapshot:
- Symbol: {live['symbol']}
- Current Price: {live['price']} {live['currency']}
- Open: {live['open']}
- Day High: {live['high']}
- Day Low: {live['low']}
- Currency: {live['currency']}
"""
        answer = generate_rag_answer(user_query, investment_mode=True, live_block=live_block)
        print("===== DEBUG END =====")
        return answer

    # Price-only questions are answered directly from Yahoo Finance.
    if ask_realtime(user_query):
        print("DEBUG: Realtime price query detected")
        symbol = extract_symbol(user_query)
        live = fetch_realtime_stock(symbol) if symbol else None
        if not live:
            print("===== DEBUG END =====")
            return "Unable to fetch the latest available price. Check the ticker symbol or try again later."
        print("===== DEBUG END =====")
        return f"""### Live Market Price: {live['symbol']}

- **Current Price:** {live['price']} {live['currency']}
- **Open:** {live['open']}
- **Day High:** {live['high']}
- **Day Low:** {live['low']}
- **Currency:** {live['currency']}

*Price data is supplied by Yahoo Finance and may be delayed.*"""

    print("DEBUG: General RAG query detected")
    answer = generate_rag_answer(user_query)
    print("===== DEBUG END =====")
    return answer


# ----------------------------------------------------
# OPTIONAL COMMAND-LINE TEST
# ----------------------------------------------------
if __name__ == "__main__":
    while True:
        question = input("\nAsk a finance question (or type 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                print("\nANSWER:\n", new_rag_ans(question))
            except Exception as exc:
                print("\nERROR:", exc)
