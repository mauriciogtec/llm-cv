from langchain_community.document_loaders import ArxivLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import hydra
from omegaconf import DictConfig
import pickle


# logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def create_splits(cfg: DictConfig):
    docs = []

    for v in cfg.arxiv.values():
        docs.extend(ArxivLoader(v).load())

    for v in cfg.pdf.values():
        docs.extend(PyPDFLoader(v).load())

    for v in cfg.web.values():
        docs.extend(WebBaseLoader(v).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=len,
    )

    splits = splitter.split_documents(docs)

    with open("data/splits.pkl", "wb") as f:
        pickle.dump(splits, f)


if __name__ == "__main__":
    create_splits()
