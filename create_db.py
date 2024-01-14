import os
import shutil
import nltk
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter


def create_db():
    nltk.download("punkt")

    arxiv_papers = {
        "AIM": "2105.13345",
    }

    papers = ArxivLoader(arxiv_papers["AIM"]).load()
    cv = PyPDFLoader("https://mauriciogtec.com/_static/cv.pdf").load()
    docs = papers + cv

    persist_directory = "./docs/chroma"
    # delete the directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embedding = OpenAIEmbeddings()
    splitter = NLTKTextSplitter(chunk_overlap=200, chunk_size=500)

    splits = splitter.split_documents(docs)

    Chroma.from_documents(
        splits,
        embedding=embedding,
        persist_directory=persist_directory,
    )


if __name__ == "__main__":
    create_db()
