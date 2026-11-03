# Campus FAQ Chatbot — Python NLP Project

## Features
- Streamlit multi-page interface with functional navigation
- FAQ retrieval using TF-IDF and optional sentence embeddings (`all-MiniLM-L6-v2`)
- Trainable Logistic Regression intent classifier
- Confidence threshold and safe fallback response
- FAQ explorer and editable JSON knowledge base
- Dashboard, query distribution, confidence charts and CSV export
- NLP laboratory: preprocessing, sentence splitting, tokenization, stop-word removal, edit distance, n-grams, PMI, POS/dependency parsing when spaCy is installed
- Text analytics: keywords, NER, temporal expression/event patterns, aspect cues and simple relation patterns
- Optional speech recognition and text-to-speech

## Setup (Windows)
1. Install Python 3.10 or 3.11 (recommended).
2. Extract the ZIP and open the `campus_faq_chatbot` folder in VS Code.
3. Open Terminal in that folder.
4. Create a virtual environment:
   `py -3.11 -m venv .venv`
5. Activate it:
   `.venv\\Scripts\\activate`
6. Install packages:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
7. Install the English spaCy model:
   `python -m spacy download en_core_web_sm`
8. Run:
   `streamlit run app.py`

The first run may download the sentence-transformer model. If that download is unavailable, the application falls back to TF-IDF retrieval.

## Important data note
`data/faqs.json` contains sample answers only. Replace these with verified information from your college's official FAQ documents, student handbook, academic regulations and placement FAQ. Do not invent dates, fees, eligibility rules or policies.

## Model design
- Intent model: TF-IDF word/bigram features + Logistic Regression, trained from the included FAQ intents.
- Semantic retrieval: `sentence-transformers/all-MiniLM-L6-v2` embeddings, when installed and downloaded.
- Baseline retrieval: TF-IDF cosine similarity.
- The confidence score is a retrieval similarity score, not a calibrated probability.

## Current scope and honest limitations
This is a runnable academic prototype. Some advanced syllabus topics (full constituency parsing, trained semantic-role labelling, TimeBank corpus training, neural machine translation and production-grade speech recognition) require additional corpora/models and are not falsely represented as completed trained components. The app includes accessible demonstrations and clearly marked extension points. The admin panel is not authenticated and is for local demonstration only.
