import os
import asyncpg
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

secret_name = "chatbot-secrets"
region_name = "us-east-1"

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY


GPT_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"

pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        min_size=1,
        max_size=5,
        timeout=10,
        ssl=False  
    )


@app.on_event("shutdown")
async def shutdown():
    await pool.close()

async def fetch_chat_history(session_id, limit=10):
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT role, content FROM chat_history
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id, limit
        )
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

async def insert_chat_message(session_id, role, content):
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO chat_history (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_id, role, content
        )

async def get_similar_chunks(embedding: list[float], top_k=5, threshold=0.7):
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT content, 1 - (embedding <#> $1::vector) AS similarity
            FROM document_chunks
            WHERE embedding <#> $1::vector < $2
            ORDER BY embedding <#> $1::vector
            LIMIT $3
            """,
            embedding, 1 - threshold, top_k
        )
        return [{"content": r["content"], "similarity": r["similarity"]} for r in rows]

# --- OpenAI Utilities ---

def get_query_embedding(query: str):
    response = openai.embeddings.create(
        model=GPT_MODEL,
        input=query
    )
    return response.data[0].embedding

def contextualize_question(chat_history, current_question):
    system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "rewrite it as a standalone question. Do NOT answer the question."
    )
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [
        {"role": "user", "content": current_question}
    ]
    response = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content.strip()


@app.post("/api/main")
async def query_handler(request: Request):
    try:
        body = await request.json()
        query_text = body.get("query", "")
        session_id = body.get("session_id", "default")
        top_k = body.get("top_k", 5)

        raw_history = await fetch_chat_history(session_id)
        chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in raw_history]

        standalone_question = contextualize_question(chat_history, query_text)
        embedding = get_query_embedding(standalone_question)

        matches = await get_similar_chunks(embedding, top_k=top_k)
        context_chunks = [r["content"] for r in matches]
        context = "\n\n".join(context_chunks) if context_chunks else "No relevant context found."

        system_prompt = (
            "You are a helpful assistant. Use the context below to answer the user's question. "
            "If the answer isn't in the context, say you don't know."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {standalone_question}"}
        ]

        completion = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages
        )
        final_answer = completion.choices[0].message.content.strip()

        await insert_chat_message(session_id, "user", query_text)
        await insert_chat_message(session_id, "assistant", final_answer)

        return {
            "answer": final_answer,
            "sources": matches 
        }

    except Exception as e:
        return {"error": str(e)}
