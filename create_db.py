import os
import shutil
import nltk
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader, BibtexLoader
from langchain.text_splitter import NLTKTextSplitter
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def create_db(cfg: DictConfig):
    nltk.download("punkt")

    docs = []

    for v in cfg.arxiv.values():
        docs.extend(ArxivLoader(v).load())

    for v in cfg.pdf.values():
        docs.extend(PyPDFLoader(v).load())

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
