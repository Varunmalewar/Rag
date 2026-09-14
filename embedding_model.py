from importlib.metadata import version
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough , RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter,TokenTextSplitter,MarkdownHeaderTextSplitter,Language
from pydantic import BaseModel, Field
from typing import List 
import tempfile
from langchain_core.documents import Document
import numpy as np

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
)



embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)


def basic_embeddings():

    # single text
    text = "What is Machine Learning?"
    single_embedding = embedding_model.embed_query(text)
    print(f"Vector dimensions: {len(single_embedding)}")
    print(f"First 100 values: {single_embedding[:100]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")


def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML.",
        "How does a neural network work?",
    ]

    batch_embedding = embedding_model.embed_documents(text)
    for i, emb in enumerate(batch_embedding):
        print(f"Text {i+1} - Vector dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")


def similarity_search():

    # Documents

    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embedding_model.embed_documents(docs)
    query_vector = embedding_model.embed_query(query)

    # compute cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")


if __name__ == "__main__":
    # basic_embeddings()
    # batch_embeddings()
    similarity_search()