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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List 
import tempfile




load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.1-flash-lite",
    api_key = SecretStr(os.environ["GOOGLE_API_KEY"]),
)



embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""



def create_kb():

    """Create a vector from knowledge base."""

    #splits the knowledge base into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50 )
    doc = Document(page_content = KNOWLEDGE_BASE , metadata = {"source": "knowledge_base"})

    chunks = splitter.split_documents([doc])

    #create vector store from chunks 

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = tempfile.mkdtemp()
    )

    return vector_store


def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k": 3})

    #Rag prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context : {context}
        Question: {question}
        Answer : 

        Make sure to answer in a concise manner and if you don't know the answe r, say "I don't know" instead of making up an answer.
        """
    )

    # format retrived documents 
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    #RAG chain (explicit RunnableParallel — a dict in a chain is implicitly RunnableParallel)
    rag_chain = (
        RunnableParallel(
            context= retriever | format_docs,
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    #Test the RAG chain with a question
    questions = [
        "What is LangChain?",
        "What are the core components of LangChain?",
        "What is LangGraph?",
        "Who  is Sachin Tendulkar?"]
    print("RAG Demo Results:")
    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Question: {question}")
        print(f"Answer: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()


