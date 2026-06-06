from functools import lru_cache

import torch
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFacePipeline,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from .config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    VECTORSTORE_DIR,
    TOP_K_RETRIEVAL,
    MAX_NEW_TOKENS,
)


@lru_cache(maxsize=1)
def load_chain():
    print("=" * 50)
    print("Loading RAG Chain...")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = (
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    print(f"Using device: {device}")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )

    print("Embeddings loaded")

    # Vector Store
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )

    print("Vectorstore loaded")

    # LLM
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if device == "cpu":
        model.to(device)

    print("Model loaded")

    hf_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(
        pipeline=hf_pipeline
    )

    prompt = PromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer cannot be found in the context,
say:

"I don't have that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    def format_docs(docs):
        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG Chain Ready")
    print("=" * 50)

    return chain


def ask_question(question: str) -> str:
    chain = load_chain()
    return chain.invoke(question)