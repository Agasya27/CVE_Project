# CVE Intelligence Analyzer — Complete Project Explanation

---

## What Is This Project?

The *CVE Intelligence Analyzer* is an AI-powered cybersecurity intelligence platform that turns raw CVE (Common Vulnerabilities and Exposures) descriptions into actionable security insight. Instead of a security analyst spending hours reading through thousands of dense NVD advisories, this tool does it in seconds — classifying the type of vulnerability, predicting its severity, summarizing it in plain English, finding historically similar vulnerabilities, and mapping it directly to your organization's technology stack.

CVEs are the global standard for publicly disclosed security vulnerabilities. The National Vulnerability Database (NVD) publishes thousands of them every year. Each one contains a technical description, a CVSS score, attack vector metadata, and affected software details. The problem is scale — there are over 200,000 CVEs published and thousands more added each month. No human team can monitor, triage, and act on all of them efficiently.

This project solves that.

---

## The Bigger Problem It Solves

### Security teams are drowning in vulnerability noise

Every week, hundreds of new CVEs are published. A typical enterprise runs hundreds of software components — web frameworks, databases, OS packages, cloud services, container runtimes. The real question is never "what vulnerabilities exist in the world?" It is always "which of these vulnerabilities affect *us*, and which one do we fix first?"

Answering that question manually requires:
- Reading every CVE description
- Mapping it to your tech stack
- Understanding the attack vector and exploitability
- Estimating business impact
- Prioritizing a patch queue for the engineering team

This is expensive, slow, and error-prone. Junior analysts miss context. Senior analysts spend time on low-severity issues. Critical vulnerabilities sit unpatched because triage takes too long.

### What this tool does instead

1. You paste a CVE description or enter your tech stack
2. The system classifies the vulnerability type, predicts severity, extracts key technical terms, and generates a plain-English summary — all in under 5 seconds
3. For your tech stack, it scans the entire CVE database (and live NVD data) and returns a risk-scored patch queue with an executive-ready security posture report
4. You can find similar historical CVEs to understand patterns and whether a class of vulnerability has been exploited before

The result is a triage workflow that takes minutes instead of days.

---

## Application Structure — Four Pages

### Page 1: CVE Analyzer
The core analysis page. You paste any CVE description and the system runs it through four AI models simultaneously:

- *Vulnerability classification* — what type of attack is this? (SQL Injection, RCE, XSS, etc.)
- *Severity prediction* — Critical, High, Medium, or Low, with confidence score
- *Keyword extraction* — the most technically significant terms from the description
- *AI summarization* — a concise plain-English summary of the threat

At the end it generates a structured security alert with a recommended remediation action based on the specific vulnerability type and severity combination.

### Page 2: Vulnerability Dashboard
An analytics dashboard over the full CVE dataset (2,311 CVEs, 1999–2024). Shows:
- Distribution of vulnerability types (bar chart)
- CVEs published per year (trend line)
- Severity breakdown (pie chart)
- Attack vector distribution (horizontal bar)
- Fully searchable and filterable CVE table with severity/type/year filters

This page answers questions like "how has the rate of SQL injection CVEs changed over time?" or "what percentage of CVEs in our dataset are Critical severity?"

### Page 3: Similar CVE Search
You describe a vulnerability in free text and the system finds the most semantically similar CVEs from the database using sentence embeddings. This is useful for:
- Understanding if a newly discovered vulnerability follows a known pattern
- Researching historical precedents when writing advisories
- Finding related patches that may apply to a new issue

Results are displayed as expandable cards with similarity scores shown as progress bars.

### Page 4: Attack Surface Scanner
The highest-value page for operational security teams. You list the technologies your organization runs (e.g., "Apache Tomcat, PostgreSQL, Spring Boot, nginx") and choose between:

- *Local scan* — instant search across the 2,311-CVE offline database
- *Live NVD scan* — real-time query of the National Vulnerability Database API for up-to-date results

The scanner then:
1. Searches for CVEs matching each technology
2. Scores each CVE by a multi-factor risk formula (CVSS base score + attack vector + exploitability + authentication requirements)
3. Ranks them into a priority patch queue (Critical → High → Medium → Low)
4. Generates an executive security posture report with patch recommendations, downloadable as CSV

This directly answers the real operational question: "given what we run, what should we patch this week?"

---

## The Models — What They Are and What They Do

### Model 1: DistilBERT Vulnerability Classifier

*What it is:* A fine-tuned transformer language model. DistilBERT is a compressed version of Google's BERT — 66 million parameters, 40% smaller and 60% faster than full BERT, retaining 97% of its performance.

*What it does:* Given a raw CVE description in natural language, it predicts which of 9 vulnerability categories the CVE belongs to:
- Buffer Overflow, CSRF, Command Injection, Cross-Site Scripting (XSS), Denial of Service, Information Disclosure, Other, Remote Code Execution, SQL Injection

*How it was trained:* Fine-tuned on 1,964 labeled CVE descriptions from the NVD. The model takes tokenized text (max 256 tokens), passes it through 6 transformer layers with multi-head self-attention, and produces a classification logit for each of the 9 categories. The category with the highest logit wins.

*Why a transformer and not something simpler:* CVE descriptions use highly technical, domain-specific language. Words like "buffer", "overflow", "arbitrary", "unauthenticated" have very different implications depending on context. TF-IDF treats each word independently. BERT understands that "execute arbitrary code" and "run malicious commands remotely" mean the same thing — it captures semantic meaning, not just keyword presence.

*Performance:*
| Metric | Value |
|---|---|
| Accuracy | 82.1% |
| Weighted F1 | 0.794 |
| SQL Injection F1 | 0.99 |
| XSS F1 | 0.96 |
| DoS F1 | 0.91 |
| RCE F1 | 0.84 |
| Command Injection F1 | 0.00 (too few samples) |

*Limitations:* Command Injection is the weakest class — only 46 total samples in the dataset, 7 in the test set. With so few examples, the model cannot reliably distinguish it from RCE and SQL Injection, whose descriptions use very similar language ("execute", "inject", "command"). More data would fix this.

---

### Model 2: Severity Classifier

*What it is:* A Logistic Regression classifier trained on TF-IDF text features combined with vulnerability-type one-hot encoding.

*What it does:* Predicts whether a CVE is Critical, High, Medium, or Low severity. This is important because CVSS scores are not always available, and even when they are, the textual description often contains contextual signals (e.g., "unauthenticated", "pre-auth", "requires physical access") that affect real-world severity assessment.

*How it works:*
1. The CVE description is cleaned and vectorized using TF-IDF with bigrams (5,000 features)
2. The vulnerability type (from Model 1 or rule-based fallback) is one-hot encoded into 13 columns
3. Both feature sets are concatenated into a 5,013-dimensional sparse vector
4. Logistic Regression with balanced class weights predicts the severity label

*Why not just use the CVSS score:* CVSS scores are not always present in the dataset, and new CVEs may not have them immediately. The text-based model works purely from description. Additionally, CVSS is a standardized formula — it does not capture organizational context or exploitability-in-the-wild signals that appear in the description text.

*Performance:*
| Metric | Value |
|---|---|
| Accuracy | 68.0% |
| Weighted F1 | 0.683 |
| 5-Fold CV Accuracy | 68.2% ± 2.9% |
| Inference latency | 0.001 ms/sample |

*Why 68%:* Severity is inherently noisy. NVD severity labels are derived mechanically from CVSS formulas, which don't always reflect real-world impact. High vs. Critical is often a matter of 0.1 points on the CVSS scale. The model is essentially learning a fuzzy boundary. 68% is reasonable for this task without the actual CVSS score available.

---

### Model 3: BART Summarizer

*What it is:* Facebook's BART (Bidirectional and Auto-Regressive Transformer) large model, pre-trained on CNN/DailyMail news summarization. Used as-is without fine-tuning.

*What it does:* Takes a verbose CVE description (often 3–6 sentences of dense technical prose) and generates a concise 1–2 sentence plain-English summary. This is the output a security analyst would write in an advisory email — what the vulnerability is, who is affected, and what the risk is.

*How it works:* BART is a sequence-to-sequence model. The encoder reads the full CVE description, the decoder generates a shorter summary token by token. The model length is bounded dynamically: max_length = min(80, word_count - 2) to avoid padding artifacts on short inputs.

*Why pre-trained (not fine-tuned):* CVE descriptions follow a similar structure to news articles — third-person factual statements about events and their consequences. BART pre-trained on news transfers well to this domain without fine-tuning. Fine-tuning would require a large dataset of (description, ideal summary) pairs, which doesn't exist publicly.

---

### Model 4: Sentence-BERT Similarity Engine

*What it is:* all-MiniLM-L6-v2 from the Sentence-Transformers library. A 22M parameter model that maps sentences to a 384-dimensional semantic embedding space.

*What it does:* Powers the Similar CVE Search page. A query description is embedded into a 384-dim vector, then compared against pre-computed embeddings for all CVEs in the database using cosine similarity. The top-K most similar CVEs are returned.

*Why this matters:* Two CVEs can describe the same class of vulnerability using completely different words. A keyword search for "heap overflow" would miss "heap-based buffer corruption" — but their sentence embeddings would be nearly identical vectors in 384-dimensional space, cosine similarity close to 1.0. Semantic similarity finds conceptually related CVEs regardless of exact wording.

*Technical detail:* All stored CVE embeddings are L2-normalized (norm = 1.0), so cosine similarity reduces to a simple dot product — fast even across thousands of vectors. The embeddings are pre-computed and stored in cve_embeddings.npy so the model only needs to embed the query at runtime.

*Current limitation:* Embeddings cover 1,314 of the 2,311 CVEs (the original dataset). The 997 CVEs added via NVD expansion are not yet embedded, meaning similarity search misses them.

---

### Model 5: TF-IDF Keyword Extractor

*What it is:* A TF-IDF (Term Frequency–Inverse Document Frequency) vectorizer fitted on the full CVE corpus.

*What it does:* Given any CVE description, it returns the top-N terms that are most statistically significant to that specific document relative to the full corpus. These are the technical keywords that define the CVE — things like "heap overflow", "arbitrary code execution", "unauthenticated remote", "integer overflow".

*Why TF-IDF works here:* A word like "vulnerability" appears in every CVE — its TF-IDF score will be near zero. A term like "use-after-free" or "type confusion" is rare across the corpus but if it appears in a specific CVE, it is almost certainly the key technical concept. TF-IDF captures exactly this.

---

## The NVD Integration

The app connects to the National Vulnerability Database REST API v2 (https://services.nvd.nist.gov/rest/json/cves/2.0) with an authenticated API key. This enables the Live NVD scan mode in the Attack Surface Scanner:

- For each technology in your stack, a keyword search is sent to NVD
- Results are paginated and fetched in real time
- CVSS scores, attack vectors, and descriptions are extracted from the JSON response
- The same risk scoring pipeline runs on live data as on the offline database

Results are cached for 1 hour per keyword to avoid redundant API calls during a session.

---

## Risk Scoring Formula

The Attack Surface Scanner does not simply sort by CVSS score. It computes a multi-factor risk score that weights:

- *CVSS base score* (0–10) — the primary severity signal
- *Attack vector* — Network-based attacks score higher than Local
- *Authentication required* — Unauthenticated attacks score higher
- *User interaction* — No interaction required scores higher
- *Confidentiality / Integrity / Availability impact* — partial vs. complete

This produces a composite risk score per CVE per technology, which drives the priority queue (Critical / High / Medium / Low patch priority labels).

---

## Fallback Architecture

The system is designed to degrade gracefully. If a heavy model fails to load (e.g., on a machine without enough RAM, or during cold start on Streamlit Cloud):

| Component | Fallback |
|---|---|
| BERT classifier | BART zero-shot classification over 8 labels |
| BART zero-shot | Rule-based keyword matching |
| Severity model | Rule-based severity from CVSS keywords |
| Sentence-BERT similarity | TF-IDF cosine similarity |
| BART summarizer | First two sentences of description |

This means the app always produces a result — it never crashes or returns an empty response. The fallback quality is lower but still useful.

---

## Data Pipeline


NVD API / raw CSV
       │
       ▼
  preprocessing.py
  ├── clean_text()           — remove HTML, special chars, normalize whitespace
  ├── preprocess_text()      — lowercase, stopword removal, lemmatization
  ├── extract_keywords()     — TF-IDF top-N terms
  └── classify_vulnerability_type()  — rule-based type detection
       │
       ▼
  data/cve_cleaned.csv
  (2,311 CVEs, 1999–2024)
       │
       ├──▶ severity_tfidf.joblib + severity_predictor.joblib  (severity model)
       ├──▶ models/bert_classifier/  (DistilBERT fine-tuned weights)
       ├──▶ tfidf_vectorizer.joblib  (keyword extraction)
       └──▶ cve_embeddings.npy       (sentence-BERT embeddings)


---

## Files and What They Do

| File | Purpose |
|---|---|
| app/streamlit_app.py | All 4 pages of the web UI |
| utils/preprocessing.py | Text cleaning, keyword extraction, rule-based classification |
| utils/model_utils.py | All model loading, inference, similarity search, alert generation |
| data/cve_cleaned.csv | Main dataset — 2,311 CVEs with labels |
| models/bert_classifier/ | DistilBERT model weights, tokenizer, config |
| models/label_encoder.joblib | Maps integer predictions back to class names (9 vuln types) |
| models/severity_predictor.joblib | Trained logistic regression severity model |
| models/severity_tfidf.joblib | TF-IDF vectorizer fitted for severity model |
| models/severity_encoder.joblib | Maps severity integers to Critical/High/Medium/Low |
| models/severity_vuln_columns.joblib | Column names for vuln-type one-hot encoding |
| models/tfidf_vectorizer.joblib | TF-IDF vectorizer for keyword extraction |
| models/cve_embeddings.npy | Pre-computed sentence-BERT embeddings (1,314 × 384) |
| .streamlit/secrets.toml | NVD API key (gitignored, stored locally and in Streamlit Cloud secrets) |
| requirements.txt | All Python dependencies including CPU-only PyTorch |

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Language models | HuggingFace Transformers (DistilBERT, BART) |
| Sentence embeddings | Sentence-Transformers (all-MiniLM-L6-v2) |
| Classical ML | scikit-learn (Logistic Regression, TF-IDF) |
| Deep learning runtime | PyTorch (CPU) |
| Data | pandas, numpy, scipy |
| Charts | Plotly |
| External API | NVD REST API v2 (authenticated) |
| Model serialization | joblib, HuggingFace safetensors |
| Deployment | Streamlit Cloud + GitHub (LFS for large model files) |

---

## What This Project Does Not Do

- *Does not replace a full SIEM or vulnerability scanner* like Tenable, Qualys, or Snyk. Those tools scan your actual systems. This tool analyzes the CVE text and maps it to your stack by keyword matching.
- *Does not verify if a CVE applies to your specific version* of a software. It matches on product name. Always cross-check with the vendor advisory.
- *Does not track exploit activity in the wild* (no integration with EPSS, GreyNoise, or Shodan).
- *Does not provide real-time alerting* — it is a dashboard, not a monitoring agent.

---

## Model Metrics Summary

| Model | Accuracy | Weighted F1 | Latency |
|---|---|---|---|
| DistilBERT (vuln type) | 82.1% | 0.794 | ~37 ms/sample |
| Severity Classifier | 68.0% | 0.683 | ~0.001 ms/sample |
| BART Summarizer | Pre-trained, no test metric | — | ~2–5 s/sample |
| Sentence-BERT Similarity | Semantic, no discrete metric | — | ~50 ms/query |
| TF-IDF Keywords | Rule-based, no test metric | — | < 1 ms/sample |
