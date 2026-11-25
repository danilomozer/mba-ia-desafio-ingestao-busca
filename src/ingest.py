import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

# O enunciado pede o PyPDFLoader do langchain-community, que hoje emite um aviso de descontinuação.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_community")
warnings.filterwarnings("ignore", message=".*langchain-community.*")
from langchain_community.document_loaders import PyPDFLoader  # noqa: E402
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import PROJECT_ROOT, ConfigError, get_embeddings, get_vector_store

load_dotenv()

# Na primeira ingestão a collection ainda não existe; esconde o aviso "Collection not found".
logging.getLogger("langchain_postgres.vectorstores").setLevel(logging.ERROR)

PDF_PATH = os.getenv("PDF_PATH")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 50


def resolve_pdf_path():
    if not PDF_PATH:
        raise ConfigError("Variável de ambiente PDF_PATH não definida. Confira o arquivo .env.")
    path = Path(PDF_PATH).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ConfigError(f"PDF não encontrado em: {path}")
    return path


def load_and_split(pdf_path):
    pages = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    chunks = splitter.split_documents(pages)

    # Mantém só metadados com valor, para não gravar campos vazios no banco.
    return [
        Document(
            page_content=chunk.page_content,
            metadata={k: v for k, v in chunk.metadata.items() if v not in ("", None)},
        )
        for chunk in chunks
    ]


def ingest_pdf():
    try:
        pdf_path = resolve_pdf_path()
        embeddings = get_embeddings()
        # Recria a collection a cada ingestão para não duplicar os chunks.
        store = get_vector_store(embeddings, pre_delete_collection=True)
    except ConfigError as exc:
        print(f"Erro de configuração: {exc}")
        sys.exit(1)

    print(f"Lendo PDF: {pdf_path}")
    chunks = load_and_split(pdf_path)
    if not chunks:
        print("Nenhum texto encontrado no PDF.")
        sys.exit(1)
    print(f"{len(chunks)} chunks gerados (tamanho {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}).")

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    for start in range(0, len(chunks), BATCH_SIZE):
        end = start + BATCH_SIZE
        store.add_documents(documents=chunks[start:end], ids=ids[start:end])
        print(f"  {min(end, len(chunks))}/{len(chunks)} chunks salvos")

    print("Ingestão concluída.")


if __name__ == "__main__":
    ingest_pdf()
