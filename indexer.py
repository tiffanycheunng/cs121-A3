
import os
import json
import hashlib
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(dict)       
        self.urls = {}                       
        self.doc_count = 0
        self.doc_lengths = defaultdict(float)

        self.stemmer = PorterStemmer() #for stemming

        # if there are duplicates 
        self.exact_hashes = set()
        self.simhashes = []

    def compute_simhash(self, tokens):
        vector = [0] * 64

        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)

            for i in range(64):
                bitmask = 1 << i
                if h & bitmask:
                    vector[i] += 1
                else:
                    vector[i] -= 1

        fingerprint = 0
        for i in range(64):
            if vector[i] > 0:
                fingerprint |= 1 << i

        return fingerprint

    def hamming_distance(self, h1, h2):
        return bin(h1 ^ h2).count("1")

    def process_file(self, filepath):

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception as e:
            return e

        url = data.get("url", "")
        html_content = data.get("content", "")
        if not html_content:
            return

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.exact_hashes:
            return
        self.exact_hashes.add(text_hash)

#making tokens
        tokens = []
        term_freq = defaultdict(float)

        for word in text.split():
            token = ''.join(c.lower() for c in word if c.isalnum())
            if token:
                stemmed = self.stemmer.stem(token)
                tokens.append(stemmed)
                term_freq[stemmed] += 1.0   
        important_tags = soup.find_all(["title", "h1", "h2", "h3", "strong", "b"])

        for tag in important_tags:
            words = tag.get_text(" ").split()
            for word in words:
                token = ''.join(c.lower() for c in word if c.isalnum())
                if token:
                    stemmed = self.stemmer.stem(token)
                    term_freq[stemmed] += 2.0  # boost weight
        simhash_value = self.compute_simhash(tokens)

        for existing_hash in self.simhashes:
            if self.hamming_distance(simhash_value, existing_hash) <= 2:
                return

        self.simhashes.append(simhash_value)

        doc_id = self.doc_count
        self.urls[doc_id] = url
        self.doc_count += 1

        for term, tf in term_freq.items():
            self.index[term][doc_id] = tf

        self.doc_lengths[doc_id] = sum(term_freq.values())

    # index entire folder
    def index_directory(self, root_path):
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".json"):
                    self.process_file(os.path.join(root, file))

    # save index to file
    def save(self, filepath):
        data = {
            "index": dict(self.index),
            "doc_lengths": dict(self.doc_lengths),
            "urls": self.urls,
            "doc_count": self.doc_count
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)