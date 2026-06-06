from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from .config import *

_rag_chain = None

def load_chain():
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K_RETRIEVAL}
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        return_full_text=False
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't have that information in the documents."
        
        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    _rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return _rag_chain


def ask_question(question: str) -> str:
    chain = load_chain()
    return chain.invoke(question)