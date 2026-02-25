import json
import math
from nltk.stem import PorterStemmer

class SearchEngine:
    def __init__(self, index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.index = data["index"]
        self.doc_lengths = data["doc_lengths"]
        self.urls = data["urls"]
        self.N = data["doc_count"]

        self.stemmer = PorterStemmer()

    def search(self, query):
        scores = {}

        og_terms = query.split()
        terms = []

        for t in og_terms:
            if t.upper() == "AND":
                continue
            token = ''.join(c.lower() for c in t if c.isalnum())
            if not token:
                continue
            stemmed = self.stemmer.stem(token)
            terms.append(stemmed)

        if not terms:
            return[]

        doc_sets = []
        for term in terms:
            if term not in self.index:
                return []
            doc_sets.append(set(self.index[term].keys()))

        candidates = set.intersection(*doc_sets)

        for term in terms:
            postings = self.index[term]
            df = len(postings)
            idf = math.log((self.N + 1) / (df + 1))
            for doc_id, tf in postings.items():
                if doc_id in candidates:
                    tfidf = tf * idf
                    scores[doc_id] = scores.get(doc_id, 0) + tfidf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:5]:
            results.append((self.urls[str(doc_id)], round(score, 4)))

        return results