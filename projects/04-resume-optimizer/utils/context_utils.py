import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def _norm(s):
    s = s.lower()
    s = re.sub(r'[^a-z0-9\+\#\.\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

@staticmethod
def extract_job_terms_tfidf(job_description, resume_text, top_k = 40):
    job = _norm(job_description)
    resume = _norm(resume_text)

    vec = TfidfVectorizer(stop_words = 'english',ngram_range = (1, 3),min_df = 1,max_df = 0.95)

    X = vec.fit_transform([job, resume])
    terms = vec.get_feature_names_out()

    job_scores = X[0].toarray().ravel()

    idx = np.argsort(job_scores)[::-1]
    ranked = [terms[i] for i in idx if job_scores[i] > 0.0]

    return ranked[:top_k]

@staticmethod
def match_terms(resume_text, job_terms):
    resume = f' {_norm(resume_text)} '
    present, missing = [], []

    for term in job_terms:
        term_norm = _norm(term)
        if not term_norm:
            continue

        if term_norm in resume:
            present.append(term)
        else:
            missing.append(term)

    return present, missing

@staticmethod
def build_context(present, missing, max_present = 15, max_missing = 20):
    present = present[:max_present]
    missing = missing[:max_missing]

    parts = []
    if present:
        parts.append('Keywords already in resume. You can use them freely: ' + ', '.join(present) + '.')
    if missing:
        parts.append('Keywords from job not found in resume, add only if applicable: ' + ', '.join(missing) + '.')

    return '\n'.join(parts)