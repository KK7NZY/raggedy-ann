# Raggady Ann

A simple Retrieval-Augmented Generation (RAG) app that lets you ask questions about a PDF using a local Ollama LLM and FAISS vector search.

## 📦 Features

- Converts PDFs to plain text
- Builds a vector store using sentence embeddings
- Retrieves relevant chunks via FAISS
- Interactive CLI session for asking questions about the book

## ⚙️ Requirements

* Python 3.13+
* uv 0.7.3+
* Ollama

## 🛠️ Setup

1. Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

2. Create a `.env` file with your Ollama host:

```env
OLLAMA_HOST=http://localhost:11434
```

3. Add your PDF file to the `assets/pdf/` directory.


## 🚀 Usage

1. Add your PDF to `assets/pdf/`.
2. Run the app:

```bash
uv run main.py --pdf-filename therustbook.pdf
````

Use `--override` to rebuild the vector store if it already exists:

```bash
uv run main.py --pdf-filename therustbook.pdf --override
```

The extracted text will be saved to `assets/txt/directory/` using the same name as the PDF (with a `.txt` extension).

The FAISS vector store will be saved to `vector_stores/<pdf_name>/`, using the PDF filename (without extension) as the folder name.


## 📚 Example Questions

* What chapters cover concurrency in Rust?
* How does borrowing work?
* Where in the book is the concept of iterators explained?
* Where are lifetimes explained?

## 🧠 Powered By

* [Ollama](https://ollama.com)
* [LangChain](https://www.langchain.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
