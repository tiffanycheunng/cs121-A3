import json
import math
from nltk.stem import PorterStemmer
from collections import defaultdict


IGNORE_URL_PATTERNS = ["?ical=1", "ical=1", "feed=rss", "feed=atom", ".ics", "calendar/export", "?replytocom="]
STOPWORDS = {"how","to","the","is","a","an","of","in","for","on","what","when","where","why"}

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
            token = ''.join(c.lower() for c in word if c.isalnum())
            if not token or token in STOPWORDS:
                continue
            stemmed = self.stemmer.stem(token)
            if stemmed in self.index:
                terms.append(stemmed)

        for i in range(len(terms) - 1):
            bigram = terms[i] + " " + terms[i + 1]
            if bigram in self.index:
                terms.append(bigram)
        
        return terms

    def search(self, query):
        terms = self.process_query(query)

        if not terms:
            return []

        query_tf = defaultdict(float)
        for term in terms:
            query_tf[term] += 1.0

        unique_terms = list(query_tf.keys())

        # Boolean AND: find common documents
        doc_sets = []
        for term in unique_terms:
            doc_sets.append(set(self.index[term].keys()))

        candidates = set.intersection(*doc_sets)

        if not candidates:
            candidates = set.union(*doc_sets)

        ### lnc.ltc implementation for cosine similarity scores
        scores = defaultdict(float)
        query_weights = {}
        query_norm_sq = 0.0

        for term in unique_terms:
            postings = self.index[term]
            q_tf = query_tf[term]
            q_wtf = 1 + math.log(q_tf, 10) if q_tf > 0 else 0

            df = len(postings)
            idf = math.log((self.N + 1) / (df + 1))

            term_weight_for_query = q_wtf * idf
            query_weights[term] = term_weight_for_query
            query_norm_sq += term_weight_for_query ** 2

        query_norm = math.sqrt(query_norm_sq)
        if query_norm == 0:
            return []

        for term in query_weights:
            query_weights[term] /= query_norm

        for term in unique_terms:
            q_weight = query_weights[term]
            postings = self.index[term]

            for doc_id, d_weight in postings.items():
                if doc_id in candidates:
                    scores[doc_id] += d_weight * q_weight

        ranked = sorted(scores.items(), key = lambda x: x[1], reverse = True)

        results = []
        for doc_id, score in ranked[:5]:
            results.append((self.urls[doc_id], round(score, 6)))
        return results
