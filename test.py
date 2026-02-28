"""Quick local smoke test for vector database retrieval."""

from llm_chains import create_embeddings, load_vectordb


def main() -> None:
    vector_db = load_vectordb(create_embeddings())
    output = vector_db.similarity_search("HoVer dataset")
    print(output)


if __name__ == "__main__":
    main()
