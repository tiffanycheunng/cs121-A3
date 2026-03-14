
import os
import json
import hashlib
from collections import defaultdict
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from nltk.stem import PorterStemmer
from math import sqrt, log
from urllib.parse import urljoin, urldefrag

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

        self.doc_norms = {}

        self.url_to_doc_id = {}
        self.raw_anchor_text = defaultdict(list)

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

        normalized_page_url = url.rstrip("/")

        warnings.filterwarnings("ignore", category = XMLParsedAsHTMLWarning)

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ", strip=True)

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.exact_hashes:
            return
        self.exact_hashes.add(text_hash)

        anchor_pairs = []                               ##anchor pairs
        for a in soup.find_all("a", href = True):
            href = a["href"].strip()
            anchor_text = a.get_text(" ", strip = True)

            if not href or not anchor_text:
                continue

            target_url = self.normalize_url(url, href)
            if target_url.startswith("http"):
                anchor_pairs.append((target_url, anchor_text))

#making tokens
        tokens = []
        term_freq = defaultdict(float)

        for word in text.split():
            stemmed = ''.join(c.lower() for c in word if c.isalnum())
            if stemmed is not None:
                tokens.append(stemmed)
                term_freq[stemmed] += 1.0

        for i in range(len(tokens) - 1):
            bigram = tokens[i] + " " + tokens[i + 1]
            term_freq[bigram] += 1.0
        
        important_tags = soup.find_all(["title", "h1", "h2", "h3", "strong", "b"])

        for tag in important_tags:
            words = tag.get_text(" ").split()
            for word in words:
                token = ''.join(c.lower() for c in word if c.isalnum())
                if token:
                    stemmed = self.stemmer.stem(token)
                    term_freq[stemmed] += 4.0  # boost weight
        simhash_value = self.compute_simhash(tokens)

        for existing_hash in self.simhashes:
            if self.hamming_distance(simhash_value, existing_hash) <= 2:
                return

        self.simhashes.append(simhash_value)

        doc_id = self.doc_count
        self.urls[doc_id] = normalized_page_url
        self.url_to_doc_id[normalized_page_url] = doc_id
        self.doc_count += 1

        self.raw_anchor_text[doc_id] = anchor_pairs

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

    def compute_doc_lnc(self):
        doc_norms = defaultdict(float)

        for term, postings in self.index.items():
            for doc_id, tf in postings.items():
                d_wtf = 1 + log(tf, 10) if tf > 0 else 0
                self.index[term][doc_id] = d_wtf
                doc_norms[doc_id] += d_wtf ** 2

        for doc_id in doc_norms:
            doc_norms[doc_id] = sqrt(doc_norms[doc_id])

        for term, postings in self.index.items():
            for doc_id, w_td in postings.items():
                norm = doc_norms[doc_id]
                if norm > 0:
                    self.index[term][doc_id] = w_td / norm
                else:
                    self.index[term][doc_id] = 0.0

        self.doc_norms = dict(doc_norms)

    def normalize_url(self, base_url, href):
        resolved = urljoin(base_url, href)
        resolved, _ = urldefrag(resolved)
        return resolved.rstrip("/")

    def tokenize_anchor_text(self, text):
        terms = []
        for word in text.split():
            stemmed = ''.join(c.lower() for c in word if c.isalnum())
            if stemmed is not None:
                terms.append(stemmed)
        return terms

    def apply_anchor_text(self, boost = 1.0):
        for source_doc_id, anchor_list in self.raw_anchor_text.items():
            for target_url, anchor_text in anchor_list:
                target_doc_id = self.url_to_doc_id.get(target_url)

                if target_doc_id is None:
                    continue

                terms = self.tokenize_anchor_text(anchor_text)

                for term in terms:
                    current = self.index[term].get(target_doc_id, 0.0)
                    self.index[term][target_doc_id] = current + boost