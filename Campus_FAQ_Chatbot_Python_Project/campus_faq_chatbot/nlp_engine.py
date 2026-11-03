
import re
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. CAMPUS FAQ KNOWLEDGE BASE
# --------------------------------------------------

FAQ_DATA = [
    {
        "question": "How can I apply for admission?",
        "answer": "Students can apply for admission through the institution's official admission portal. Check the eligibility requirements, complete the application form, and submit the required documents.",
        "category": "Admissions"
    },
    {
        "question": "What documents are required for admission?",
        "answer": "Commonly required documents include academic certificates, transfer certificate, identification proof, passport-size photographs, and other documents specified by the admissions office.",
        "category": "Admissions"
    },
    {
        "question": "How do I pay my college fees?",
        "answer": "Fee payment procedures and deadlines are generally available through the student portal or accounts office. Please verify the current payment instructions with the institution.",
        "category": "Fees"
    },
    {
        "question": "Is there a scholarship facility?",
        "answer": "Scholarships may be available based on academic performance, eligibility criteria, and applicable government or institutional schemes. Contact the scholarship office for current details.",
        "category": "Fees"
    },
    {
        "question": "Where is the library located?",
        "answer": "The campus library provides access to academic books, journals, reference materials, and digital learning resources. Check the campus directory for its exact location and timings.",
        "category": "Facilities"
    },
    {
        "question": "What are the library timings?",
        "answer": "Library opening hours may vary during regular working days, examinations, and holidays. Please confirm the current schedule with the library.",
        "category": "Facilities"
    },
    {
        "question": "Is Wi-Fi available on campus?",
        "answer": "Campus Wi-Fi may be available for students and staff. Contact the IT support team for network access and login instructions.",
        "category": "Facilities"
    },
    {
        "question": "How can I contact the placement cell?",
        "answer": "Students can contact the placement and training department for placement registration, recruitment schedules, company details, and career guidance.",
        "category": "Placements"
    },
    {
        "question": "Are internships provided?",
        "answer": "Internship opportunities may be shared through departments and the placement cell. Students should check official announcements and eligibility requirements.",
        "category": "Placements"
    },
    {
        "question": "What is the attendance requirement?",
        "answer": "Attendance requirements are determined by institutional and university regulations. Students should consult their department or academic office for the applicable minimum.",
        "category": "Academics"
    },
    {
        "question": "How can I get my exam timetable?",
        "answer": "Examination timetables are generally published through the examination cell, department notice boards, or official student portal.",
        "category": "Examinations"
    },
    {
        "question": "How do I apply for a leave?",
        "answer": "Students should follow their department's leave application procedure and submit the required details and supporting documents to the appropriate authority.",
        "category": "Academics"
    },
    {
        "question": "Are sports facilities available?",
        "answer": "Sports and physical education facilities may be available for students. Contact the sports department for facility details and activity schedules.",
        "category": "Facilities"
    },
    {
        "question": "Is hostel accommodation available?",
        "answer": "Hostel availability, application procedures, fees, and rules should be confirmed with the institution's hostel administration.",
        "category": "Accommodation"
    },
    {
        "question": "How can I get a bonafide certificate?",
        "answer": "Students can request a bonafide certificate through the administrative office or the institution's student service procedure.",
        "category": "Administration"
    },
    {
        "question": "Who should I contact for technical support?",
        "answer": "For technical issues involving student portals, Wi-Fi, or computer facilities, contact the campus IT support team.",
        "category": "Support"
    },
    {
        "question": "What courses are offered?",
        "answer": "Available programmes and course details should be checked in the institution's official prospectus or admissions portal.",
        "category": "Admissions"
    },
    {
        "question": "How can I get a transfer certificate?",
        "answer": "Students should contact the administrative office and follow the official transfer certificate application process.",
        "category": "Administration"
    }
]


# --------------------------------------------------
# 2. TEXT PREPROCESSING
# --------------------------------------------------

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


def sentence_segment(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", str(text))
        if sentence.strip()
    ]


# --------------------------------------------------
# 3. SYLLABUS MAPPING READER
# --------------------------------------------------

def load_syllabus_mapping():
    path = Path(__file__).with_name("syllabus_mapping.md")

    if not path.exists():
        return []

    rows = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line.startswith("|"):
            continue

        cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]

        if len(cells) < 3:
            continue

        if cells[0].lower() == "unit" or set(cells[0]) <= {"-", ":"}:
            continue

        if len(cells) >= 3 and cells[0] in {"I", "II", "III", "IV", "V"}:
            rows.append({
                "unit": cells[0],
                "topic": cells[1],
                "implementation": cells[2]
            })

    return rows


# --------------------------------------------------
# 4. TF-IDF CHATBOT AND RETRIEVAL
# --------------------------------------------------

class CampusChatbot:

    def __init__(self):
        self.faqs = FAQ_DATA
        self.mapping = load_syllabus_mapping()

        self.documents = [
            preprocess_text(item["question"])
            for item in self.faqs
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )

        self.matrix = self.vectorizer.fit_transform(self.documents)

    def get_response(self, question):
        question = str(question).strip()

        if not question:
            return {
                "answer": "Please enter a question.",
                "confidence": 0.0,
                "category": "Invalid"
            }

        query = preprocess_text(question)

        if not query:
            return {
                "answer": "Please enter a meaningful question.",
                "confidence": 0.0,
                "category": "Invalid"
            }

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]

        best_index = int(np.argmax(scores))
        confidence = float(scores[best_index])

        if confidence >= 0.12:
            faq = self.faqs[best_index]

            return {
                "answer": faq["answer"],
                "confidence": confidence,
                "category": faq["category"]
            }

        # Search the syllabus mapping if no FAQ matches well.
        mapping_documents = [
            preprocess_text(
                row["topic"] + " " + row["implementation"]
            )
            for row in self.mapping
        ]

        if mapping_documents:
            mapping_matrix = self.vectorizer.transform(
                mapping_documents
            )
            mapping_scores = cosine_similarity(
                query_vector, mapping_matrix
            )[0]

            mapping_index = int(np.argmax(mapping_scores))
            mapping_confidence = float(mapping_scores[mapping_index])

            if mapping_confidence >= 0.12:
                row = self.mapping[mapping_index]

                return {
                    "answer": (
                        f"Unit {row['unit']} – {row['topic']}\n\n"
                        f"Project implementation: {row['implementation']}"
                    ),
                    "confidence": mapping_confidence,
                    "category": "Syllabus Mapping"
                }

        return {
            "answer": (
                "I could not find a sufficiently relevant answer. "
                "Please try asking about admissions, fees, academics, "
                "examinations, facilities, placements, or administration."
            ),
            "confidence": confidence,
            "category": "Unknown"
        }


chatbot = CampusChatbot()


def get_answer(question):
    return chatbot.get_response(question)


# --------------------------------------------------
# 5. MINIMUM EDIT DISTANCE
# --------------------------------------------------

def levenshtein_distance(first, second):
    first = str(first)
    second = str(second)

    rows = len(first) + 1
    cols = len(second) + 1

    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i

    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if first[i - 1] == second[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[-1][-1]


# --------------------------------------------------
# 6. N-GRAM LANGUAGE MODEL
# --------------------------------------------------

def get_ngram_analysis(text):
    words = tokenize(text)

    unigrams = Counter(words)
    bigrams = Counter(zip(words, words[1:]))

    return {
        "tokens": words,
        "unigrams": unigrams,
        "bigrams": bigrams
    }


def laplace_bigram_probability(first, second, words):
    bigrams = Counter(zip(words, words[1:]))
    unigrams = Counter(words)

    vocabulary_size = max(len(set(words)), 1)

    numerator = bigrams[(first, second)] + 1
    denominator = unigrams[first] + vocabulary_size

    return numerator / denominator


def calculate_perplexity(text):
    words = tokenize(text)

    if len(words) < 2:
        return None

    bigrams = Counter(zip(words, words[1:]))
    unigrams = Counter(words)
    vocabulary_size = len(set(words))

    log_probability = 0

    for first, second in zip(words, words[1:]):
        probability = (
            bigrams[(first, second)] + 1
        ) / (
            unigrams[first] + vocabulary_size
        )

        log_probability += math.log(probability)

    return math.exp(-log_probability / (len(words) - 1))


# --------------------------------------------------
# 7. POINTWISE MUTUAL INFORMATION
# --------------------------------------------------

def calculate_pmi(text):
    words = tokenize(text)

    if len(words) < 2:
        return []

    unigram_counts = Counter(words)
    bigram_counts = Counter(zip(words, words[1:]))

    total_words = len(words)
    total_bigrams = len(words) - 1

    results = []

    for (first, second), count in bigram_counts.items():
        p_xy = count / total_bigrams
        p_x = unigram_counts[first] / total_words
        p_y = unigram_counts[second] / total_words

        pmi = math.log2(p_xy / (p_x * p_y))

        results.append({
            "Word Pair": f"{first} → {second}",
            "PMI": round(pmi, 4),
            "Frequency": count
        })

    return sorted(
        results,
        key=lambda item: item["PMI"],
        reverse=True
    )


# --------------------------------------------------
# 8. OPTIONAL SPACY NLP
# --------------------------------------------------

def load_spacy_model():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm"), "en_core_web_sm"
        except OSError:
            return spacy.blank("en"), "Blank English pipeline"

    except ImportError:
        return None, "spaCy not installed"


def analyze_with_spacy(text):
    nlp, model_name = load_spacy_model()

    if nlp is None:
        return {
            "model": model_name,
            "tokens": [],
            "entities": [],
            "dependencies": [],
            "pos_available": False
        }

    doc = nlp(str(text))

    has_parser = "parser" in nlp.pipe_names
    has_tagger = "tagger" in nlp.pipe_names or "morphologizer" in nlp.pipe_names
    has_ner = "ner" in nlp.pipe_names

    return {
        "model": model_name,
        "tokens": [
            {
                "Token": token.text,
                "POS": token.pos_ if has_tagger else "Unavailable",
                "Lemma": token.lemma_ if token.lemma_ else token.text
            }
            for token in doc
        ],
        "entities": [
            {
                "Entity": entity.text,
                "Label": entity.label_
            }
            for entity in doc.ents
        ],
        "dependencies": [
            {
                "Token": token.text,
                "Head": token.head.text,
                "Dependency": token.dep_
            }
            for token in doc
        ] if has_parser else [],
        "pos_available": has_tagger,
        "ner_available": has_ner,
        "parser_available": has_parser
    }


# --------------------------------------------------
# 9. RULE-BASED ENTITY RECOGNITION
# --------------------------------------------------

def rule_based_ner(text):
    patterns = {
        "EMAIL": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"\b(?:\+?\d[\d\s-]{7,}\d)\b",
        "DATE": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        "TIME": r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b"
    }

    results = []

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, str(text)):
            results.append({
                "Entity": match.group(),
                "Label": label
            })

    return results


# --------------------------------------------------
# 10. RELATION, EVENT AND TEMPORAL EXTRACTION
# --------------------------------------------------

def extract_relations(text):
    patterns = [
        (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+works at\s+([A-Z][\w ]+)",
         "WORKS_AT"),
        (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+studies at\s+([A-Z][\w ]+)",
         "STUDIES_AT"),
        (r"(\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+joined\s+([A-Z][\w ]+)",
         "JOINED")
    ]

    results = []

    for pattern, relation in patterns:
        for match in re.finditer(pattern, str(text)):
            results.append({
                "Subject": match.group(1).strip(),
                "Relation": relation,
                "Object": match.group(2).strip()
            })

    return results


def extract_events_and_time(text):
    event_words = [
        "admission", "exam", "examination", "meeting",
        "registration", "orientation", "workshop",
        "competition", "interview", "submission"
    ]

    temporal_patterns = {
        "DATE": r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
        "TIME": r"\b\d{1,2}(?::\d{2})?\s?(?:AM|PM|am|pm)\b",
        "RELATIVE_TIME": r"\b(today|tomorrow|yesterday|next week|next month|last week)\b"
    }

    lower_text = str(text).lower()

    events = [
        word for word in event_words
        if re.search(r"\b" + re.escape(word) + r"\b", lower_text)
    ]

    temporal = []

    for label, pattern in temporal_patterns.items():
        for match in re.finditer(pattern, str(text), re.IGNORECASE):
            temporal.append({
                "Expression": match.group(),
                "Type": label
            })

    return events, temporal


# --------------------------------------------------
# 11. ASPECT AND TEMPORAL ANALYSIS
# --------------------------------------------------

ASPECT_CUES = {
    "Teaching": ["teacher", "faculty", "teaching", "lecture"],
    "Infrastructure": ["building", "classroom", "laboratory", "lab", "wifi"],
    "Administration": ["office", "administration", "certificate", "staff"],
    "Facilities": ["library", "hostel", "canteen", "sports", "transport"],
    "Academics": ["exam", "course", "syllabus", "attendance", "class"]
}

def analyze_aspects(text):
    words = set(tokenize(text))
    results = []

    for aspect, cues in ASPECT_CUES.items():
        matched = words.intersection(cues)

        if matched:
            results.append({
                "Aspect": aspect,
                "Matched Terms": ", ".join(sorted(matched)),
                "Count": len(matched)
            })

    positive = {"good", "excellent", "helpful", "great", "clean", "fast"}
    negative = {"bad", "poor", "slow", "difficult", "worst", "unhelpful"}

    positive_count = len(words.intersection(positive))
    negative_count = len(words.intersection(negative))

    if positive_count > negative_count:
        sentiment = "Positive"
    elif negative_count > positive_count:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return results, sentiment


# --------------------------------------------------
# 12. SIMPLE SEMANTIC ROLE DEMONSTRATION
# --------------------------------------------------

def demonstrate_srl(text):
    words = tokenize(text)

    if not words:
        return []

    action_words = {
        "submitted", "attended", "joined", "completed",
        "conducted", "organized", "registered", "visited"
    }

    action_index = next(
        (i for i, word in enumerate(words) if word in action_words),
        None
    )

    if action_index is None:
        return [{
            "Role": "Information",
            "Text": str(text),
            "Method": "Rule-based demonstration"
        }]

    return [
        {
            "Role": "Possible Agent",
            "Text": " ".join(words[:action_index]) or "Not identified",
            "Method": "Rule-based demonstration"
        },
        {
            "Role": "Predicate",
            "Text": words[action_index],
            "Method": "Rule-based demonstration"
        },
        {
            "Role": "Possible Context / Object",
            "Text": " ".join(words[action_index + 1:]) or "Not identified",
            "Method": "Rule-based demonstration"
        }
    ]


# --------------------------------------------------
# 13. OPTIONAL SEMANTIC EMBEDDINGS
# --------------------------------------------------

def semantic_similarity(question, candidate):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = model.encode(
            [question, candidate],
            convert_to_numpy=True
        )

        score = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]

        return float(score), "SentenceTransformer MiniLM"

    except Exception:
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([question, candidate])

        score = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]

        return float(score), "TF-IDF fallback"


# --------------------------------------------------
# 14. PROJECT INFORMATION
# --------------------------------------------------

def get_project_statistics():
    categories = Counter(item["category"] for item in FAQ_DATA)

    return {
        "faq_count": len(FAQ_DATA),
        "category_count": len(categories),
        "categories": dict(categories),
        "syllabus_topics": len(load_syllabus_mapping())
    }
