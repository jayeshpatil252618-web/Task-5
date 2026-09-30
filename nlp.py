import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "the","and","for","with","that","this","from","are","you","your",
    "our","will","have","has","had","was","were","been","being","into",
    "using","use","used","about","their","they","them","his","her",
    "its","who","what","where","when","how","all","any","can","should",
    "must","not","but","also","than","then","there","here","job","work",
    "working","role","team","years","year","experience","skills"
}

def clean_words(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower())

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 2
    ]

def keyword_analysis(resume_text, job_text):
    resume_words = set(clean_words(resume_text))
    job_words = clean_words(job_text)

    counts = Counter(job_words)

    important = [
        word for word, count in counts.most_common(30)
    ]

    matched = [
        word for word in important
        if word in resume_words
    ]

    missing = [
        word for word in important
        if word not in resume_words
    ]

    return matched[:15], missing[:15]

def calculate_match(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        [resume_text, job_text]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return float(similarity * 100)
