import json
import math
from nltk.stem import PorterStemmer


IGNORE_URL_PATTERNS = ["?ical=1", "ical=1", "feed=rss", "feed=atom", ".ics", "calendar/export", "?replytocom="]


class SearchEngine:
    def __init__(self, index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Convert keys back to int (JSON stores them as strings)
        self.index = {
            term: {int(doc_id): tf for doc_id, tf in postings.items()}
            for term, postings in data["index"].items()
        }
        self.doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
        self.urls = {int(k): v for k, v in data["urls"].items()}
        self.N = data["doc_count"]
        self.stemmer = PorterStemmer()

    def is_junk_url(self, url):
        return any(pattern in url for pattern in IGNORE_URL_PATTERNS)

    def process_query(self, query):
        terms = []
        for word in query.split():
            if word.upper() == "AND":
                continue
            token = ''.join(c.lower() for c in word if c.isalnum())
            if not token:
                continue
            stemmed = self.stemmer.stem(token)
            if stemmed in self.index:
                terms.append(stemmed)
        return terms

    def search(self, query):
        terms = self.process_query(query)
        if not terms:
            return []

        # Boolean AND: find common documents
        doc_sets = []
        for term in terms:
            doc_sets.append(set(self.index[term].keys()))

        candidates = set.intersection(*doc_sets)

        if not candidates:
            candidates = set.union(*doc_sets)

        scores = {}
        for term in terms:
            postings = self.index[term]
            df = len(postings)
            idf = math.log((self.N + 1) / (df + 1))
            for doc_id in candidates:
                if doc_id in postings:
                    tf = postings[doc_id]
                    normalized_score = (tf * idf) / self.doc_lengths[doc_id]
                    scores[doc_id] = scores.get(doc_id, 0) + normalized_score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Keep original junk URL filter, return top 5 per teammate's version
        results = []
        for doc_id, score in ranked:
            url = self.urls[doc_id]
            if not self.is_junk_url(url):
                results.append((url, round(score, 6)))
            if len(results) == 5:
                break

        return results