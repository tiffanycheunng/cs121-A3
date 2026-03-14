import os
from indexer import InvertedIndex
from search import SearchEngine


def build_index():
    index = InvertedIndex()
    index.index_directory("ANALYST")
    index.apply_anchor_text(boost = 1.0)
    index.compute_doc_lnc()
    index.save("index.json")
    print("doc_count:", index.doc_count)
    print("Index built successfully!")
    print("Indexed documents:", index.doc_count)
    print("Unique tokens:", len(index.index))
    print("Index size (KB):", round(os.path.getsize("index.json") / 1024, 2))

def run_search():
    engine = SearchEngine("index.json")
    while True:
        query = input("\nEnter query (or type exit_out): ")

        if query.lower() == "exit_out":
            break
        results = engine.search(query)
        if not results:
            print("\nNo relevant results found. The database may not contain pages related to this query.")
        else:
            print("\nTop Results:")
            for url, score in results:
                print(url, "| score:", score)


if __name__ == "__main__":
    build_index()
    run_search()