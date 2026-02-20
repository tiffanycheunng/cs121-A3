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

        terms = query.split()

        for term in terms:
            token = ''.join(c.lower() for c in term if c.isalnum())
            if not token:
                continue

            stemmed = self.stemmer.stem(token)

            if stemmed not in self.index:
                continue

            df = len(self.index[stemmed])
            idf = math.log((self.N + 1) / (df + 1))

            for doc_id, tf in self.index[stemmed].items():
                tfidf = tf * idf
                scores[doc_id] = scores.get(doc_id, 0) + tfidf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:10]:
            results.append((self.urls[str(doc_id)], round(score, 4)))

        return results