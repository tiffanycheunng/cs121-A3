import os
from indexer import InvertedIndex
from search import SearchEngine

def build_index():
    index = InvertedIndex()
    index.index_directory("ANALYST")   
    index.save("index.json")

    print("doc_count:", index.doc_count)
    print("Index built successfully!")
    print("Indexed documents:", index.doc_count)
    print("Unique tokens:", len(index.index))
    print("Index size (KB):", round(os.path.getsize("index.json") / 1024, 2))
    
def test_required_queries():
    engine = SearchEngine("index.json")

    queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering"
    ]

    for q in queries:
        print("\nQuery:", q)
        results = engine.search(q)

        for url, score in results:
            print(url, "| score:", score)

def run_search():
    engine = SearchEngine("index.json")

    while True:
        query = input("\nEnter query (or type exit): ")

        if query.lower() == "exit":
            break

        results = engine.search(query)

        print("\nTop Results:")
        for url, score in results:
            print(url, "| score:", score)

if __name__ == "__main__":
    build_index()
    test_required_queries()
    run_search()
