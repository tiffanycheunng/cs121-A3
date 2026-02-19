
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
        except Exception:
            return  

        url = data.get("url", "")
        html_content = data.get("content", "")

        if not html_content:
            return

        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(" ")

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.exact_hashes:
            return
        self.exact_hashes.add(text_hash)

#making tokens
        tokens = []
        term_freq = defaultdict(float)

        words = text.split()

        for word in words:
            token = ''.join(c.lower() for c in word if c.isalnum())
            if token:
                stemmed = self.stemmer.stem(token)
                tokens.append(stemmed)
                term_freq[stemmed] += 1.0


