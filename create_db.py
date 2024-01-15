import os
import shutil
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import ArxivLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hydra
from omegaconf import DictConfig
import logging


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def create_db(cfg: DictConfig):
    docs = []

    for v in cfg.arxiv.values():
        docs.extend(ArxivLoader(v).load())

    for v in cfg.pdf.values():
        docs.extend(PyPDFLoader(v).load())

    for v in cfg.web.values():
        docs.extend(WebBaseLoader(v).load())

    persist_directory = "./docs/chroma"
    # delete the directory if it exists
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embedding = OpenAIEmbeddings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
    )

    splits = splitter.split_documents(docs)

    db = Chroma.from_documents(
        splits,
        embedding=embedding,
        persist_directory=persist_directory,
    )
    # try:
    #     db.persist()
    # except Exception as e:
    #     logging.error(f"Error persisting db: {e}")

    logging.info(f"Created db with {len(splits)} documents")

    return db


if __name__ == "__main__":
    create_db()
