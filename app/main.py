import traceback

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag import ask_question
from .rag import load_chain


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting application...")
        load_chain()
        print("Application ready")
    except Exception as e:
        print("Startup failed:")
        traceback.print_exc()
        raise e

    yield

    print("Application shutting down...")


app = FastAPI(
    title="RAG Chatbot API",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def root():
    return {
        "status": "RAG Chatbot API is running"
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
async def chat(req: ChatRequest):

    if not req.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        answer = ask_question(
            req.question
        )

        return ChatResponse(
            answer=answer
        )

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )