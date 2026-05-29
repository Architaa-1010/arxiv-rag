import os
import json
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="arxiv_smart",
    metadata={"hnsw:space": "cosine"}
)
print(f"Ready. {collection.count()} chunks loaded.\n")

TEST_QUESTIONS = [
    "What methods are used to reduce hallucination in language models?",
    "How does retrieval augmented generation improve question answering?",
    "What is chain of thought reasoning and how does it improve LLM performance?",
    "How does reinforcement learning from human feedback work?",
    "What evaluation metrics are used to assess RAG systems?",
    "How do cross-encoders differ from bi-encoders in retrieval?",
    "What are the main challenges in fine-tuning large language models?",
    "How do diffusion models generate images?",
    "What is instruction tuning and why is it important?",
    "How does knowledge distillation compress large models?",
]


def retrieve_and_rerank(query, n_results=5):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=20
    )
    candidates = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        candidates.append({
            "text": doc,
            "title": meta["title"],
            "section": meta.get("section", "unknown")
        })
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for i, c in enumerate(candidates):
        c["rerank_score"] = float(scores[i])
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:n_results]


def generate_answer(question, chunks):
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"Source [{i+1}]: {chunk['text']}\n---\n"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are a research assistant. Answer using ONLY the provided sources.
Cite sources like [1], [2]. If sources lack info, say so honestly.

Sources:\n""" + context
            },
            {"role": "user", "content": question}
        ],
        max_tokens=512,
        temperature=0.1
    )
    return response.choices[0].message.content


def score_faithfulness(question, answer, chunks):
    """Ask LLM: is every claim in the answer supported by the sources?"""
    context = "\n---\n".join([c["text"] for c in chunks])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Your job is to score how faithful an answer is to its sources.

Faithfulness means: every claim in the answer must be directly supported by the provided source chunks.
Score from 0.0 to 1.0 where:
- 1.0 = every claim is fully supported by the sources
- 0.5 = about half the claims are supported
- 0.0 = answer ignores the sources completely

Respond with ONLY a JSON object like this:
{"score": 0.85, "reason": "one sentence explanation"}"""
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nANSWER: {answer}\n\nSOURCE CHUNKS:\n{context[:3000]}"
            }
        ],
        max_tokens=150,
        temperature=0
    )

    try:
        text = response.choices[0].message.content.strip()
        # extract json even if there's extra text
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        return float(result["score"]), result.get("reason", "")
    except Exception:
        return 0.5, "parse error"


def score_relevancy(question, answer):
    """Ask LLM: does the answer actually address the question?"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Score how relevant an answer is to the question.

Relevancy means: does the answer directly address what was asked?
Score from 0.0 to 1.0 where:
- 1.0 = answer directly and completely addresses the question
- 0.5 = answer partially addresses the question
- 0.0 = answer is off-topic

Respond with ONLY a JSON object like this:
{"score": 0.85, "reason": "one sentence explanation"}"""
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nANSWER: {answer}"
            }
        ],
        max_tokens=150,
        temperature=0
    )

    try:
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        return float(result["score"]), result.get("reason", "")
    except Exception:
        return 0.5, "parse error"


def score_context_quality(question, chunks):
    """Ask LLM: were the retrieved chunks actually relevant to the question?"""
    context_titles = "\n".join([
        f"[{i+1}] {c['title'][:60]} (section: {c['section']})"
        for i, c in enumerate(chunks)
    ])

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """You are an evaluation judge. Score the quality of retrieved context for a question.

Context quality means: are the retrieved chunks relevant and useful for answering the question?
Score from 0.0 to 1.0 where:
- 1.0 = all retrieved chunks are highly relevant
- 0.5 = some chunks are relevant, some are off-topic
- 0.0 = retrieved chunks are completely irrelevant

Respond with ONLY a JSON object like this:
{"score": 0.85, "reason": "one sentence explanation"}"""
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nRETRIEVED CHUNKS:\n{context_titles}"
            }
        ],
        max_tokens=150,
        temperature=0
    )

    try:
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        result = json.loads(text[start:end])
        return float(result["score"]), result.get("reason", "")
    except Exception:
        return 0.5, "parse error"


def run_evaluation():
    print("Running custom RAG evaluation on 10 questions...")
    print("="*60)

    all_results = []
    total_faith = 0
    total_relev = 0
    total_ctx = 0

    for i, question in enumerate(TEST_QUESTIONS):
        print(f"\n[{i+1}/10] {question[:65]}...")

        chunks = retrieve_and_rerank(question)
        answer = generate_answer(question, chunks)

        faith_score, faith_reason = score_faithfulness(question, answer, chunks)
        relev_score, relev_reason = score_relevancy(question, answer)
        ctx_score, ctx_reason = score_context_quality(question, chunks)

        total_faith += faith_score
        total_relev += relev_score
        total_ctx += ctx_score

        print(f"  Faithfulness    : {faith_score:.2f} — {faith_reason[:70]}")
        print(f"  Answer Relevancy: {relev_score:.2f} — {relev_reason[:70]}")
        print(f"  Context Quality : {ctx_score:.2f} — {ctx_reason[:70]}")

        all_results.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith_score,
            "answer_relevancy": relev_score,
            "context_quality": ctx_score,
            "faithfulness_reason": faith_reason,
            "relevancy_reason": relev_reason,
            "context_reason": ctx_reason
        })

    n = len(TEST_QUESTIONS)
    avg_faith = total_faith / n
    avg_relev = total_relev / n
    avg_ctx = total_ctx / n
    overall = (avg_faith + avg_relev + avg_ctx) / 3

    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print(f"  Faithfulness     : {avg_faith:.3f}")
    print(f"  Answer Relevancy : {avg_relev:.3f}")
    print(f"  Context Quality  : {avg_ctx:.3f}")
    print(f"  Overall Average  : {overall:.3f}")
    print("="*60)

    # save to file
    output = {
        "summary": {
            "faithfulness": round(avg_faith, 3),
            "answer_relevancy": round(avg_relev, 3),
            "context_quality": round(avg_ctx, 3),
            "overall": round(overall, 3)
        },
        "per_question": all_results
    }

    with open("eval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nFull results saved to eval_results.json")
    print("\nThese scores can go directly on your resume and README!")
    return output


if __name__ == "__main__":
    run_evaluation()