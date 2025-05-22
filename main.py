import argparse
import os
import logging
from pathlib import Path

import ollama
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# Silence httpx info-level logs
logging.getLogger("httpx").setLevel(logging.WARNING)


def pdf_to_text(filepath: str) -> Path:
    """
    Converts a PDF file to text.

    :filepath: Path to the PDF file to convert to text.
    :return: None
    """
    if not isinstance(filepath, str):
        raise TypeError("filepath must be a string")

    filepath = filepath.strip()

    if not filepath:
        raise ValueError("filepath cannot be empty")

    input_path = Path(filepath)

    try:
        input_path = input_path.relative_to(Path.cwd())
    except ValueError:
        pass

    output_dir = Path("./assets/txt/")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.txt"

    if output_path.exists():
        logger.debug(f"File {output_path} already exists. Returning early.")
        return output_path

    logger.info(f"Converting PDF {filepath} to text")

    reader = PdfReader(input_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, page in enumerate(reader.pages):
            try:
                f.write(page.extract_text() + "\n")
            except Exception as e:
                logger.error(f"Error extracting text from page {i}: {e}")

    logger.info(f"PDF {input_path} converted to text file {output_path}")

    return output_path


def build_vector_store(text_filepath: Path, chunk_size: int = 1000, chunk_overlap: int = 100, override: bool = False) -> FAISS:
    """
    Build vector store from text file.

    :param text_filepath: Path to the text file to build the vector store from.
    :param chunk_size: Size of each chunk to split the text into.
    :param chunk_overlap: Overlap between chunks.
    :return: Vector store
    """
    if not text_filepath.exists():
        raise ValueError(f"Text file {text_filepath} does not exist.")

    # Ensure the output directory exists
    db_dir = Path("./store") / text_filepath.stem

    # If vector store already exists and override is false, raise an error
    if db_dir.exists() and not override:
        raise FileExistsError(f"Vector store already exists at '{str(db_dir)}'. Use override=True to rebuild.")

    logger.info(f"Building vector store, {chunk_size=}, {chunk_overlap=}")

    db_dir.mkdir(parents=True, exist_ok=True)

    # Load the text from file.
    text = text_filepath.read_text(encoding="utf-8")

    # Split the text into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = [Document(chunk) for chunk in text_splitter.split_text(text)]

    # Create an embedding model that converts text into vectors for semantic search.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store from documents using FAISS (by Facebook), an in-memory vector store
    # supposedly faster than Chroma for local search.
    vector_store = FAISS.from_documents(documents, embeddings)

    # Save the vector store to disk
    vector_store.save_local(str(db_dir))

    logger.info(f"Vector store saved to '{db_dir.resolve()}'")

    return vector_store


def load_vector_store(text_filepath: Path) -> FAISS:
    """
    Load vector store file.

    :param text_filepath: Path to the text file to build the vector store from.
    :return: Vector store
    """
    db_dir = Path("./store") / text_filepath.stem

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_store = FAISS.load_local(str(db_dir), embeddings, allow_dangerous_deserialization=True)

    logger.info(f"{db_dir.resolve()} vector store loaded")

    return vector_store


def init_rag_session(vector_store, model: str = "mistral-nemo:latest", k: int = 4) -> None:
    """
    Starts an interactive RAG session using an Ollama model and a FAISS vector store.

    Note: Connects to the Ollama server specified in the .env file. Retrieves top-k relevant chunks
    and generates responses using one-off prompts.

    :param vector_store: FAISS vector store containing embedded document chunks.
    :param model: Name of the Ollama model to use for generation. Defaults to "mistral-nemo:latest".
    :param k: Number of top document chunks to retrieve for each query. Defaults to 4.
    :return: None
    """
    print("=" * 43)
    print("=" * 43)
    print("💬 RAG Session started. Type '/bye' to quit.")
    print("=" * 43)

    prompt_template = """\
    Use the context below to give a short, clear, and concise answer.
    Limit your answer to 2–3 sentences.

    Context: {context}
    Question: {question}
    Answer:"""

    while query := input("> ").strip():
        if query.lower() == "/bye":
            break

        # Retrieve top-k relevant chunks
        docs = vector_store.similarity_search(query, k=k)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Format the prompt template
        prompt = prompt_template.format(context=context, question=query)

        try:
            response = ollama.generate(model=model, prompt=prompt)
            response = response["response"].strip()
            print(f"> {response}")
        except Exception as e:
            print(f"> Exception raised: {e}")

    print("👋 Exiting RAG session. Goodbye!")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Example using FAISS and Ollama")
    parser.add_argument("--pdf-filename", required=True, help="PDF filename (must exist in 'assets/pdf/')")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of characters per chunk (default: 1000)")
    parser.add_argument(
        "--chunk-overlap", type=int, default=100, help="Number of overlapping characters between chunks (default: 100)"
    )
    parser.add_argument("--override", action="store_true", help="Override existing vector store if it exists", default=False)
    return parser.parse_args()


def main():
    logger.info("Preparing to embed text")

    args = parse_args()

    # Convert PDF to text
    text_path = pdf_to_text(f"./assets/pdf/{args.pdf_filename}.pdf")

    # Build vector store from text
    try:
        vector_store = build_vector_store(
            text_path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, override=args.override
        )
    except FileExistsError:
        logger.warning(f"File '{text_path}' already exists. Use override=True to rebuild.")
        vector_store = load_vector_store(text_path)

    index = vector_store.index

    if index is None:
        raise ValueError("Vector store index is None")

    # Log vector count and dimension
    logger.info(f"FAISS index has {index.ntotal} vectors of dimension {index.d}.")

    # Start RAG Session
    init_rag_session(vector_store)


if __name__ == "__main__":
    main()
