"""Deterministic generator for the `temporal_mvp` benchmark (Goal 6, MVP).

Builds a small, controlled "enterprise knowledge base" where several facts have
multiple time-stamped versions (old -> new). Produces two files that plug
straight into the existing pipeline (no changes to main.py):

    reproduce/dataset/temporal_mvp_corpus.json   # retrieval corpus
    reproduce/dataset/temporal_mvp.json          # questions + gold

Schema (additive over the existing corpus format; extra fields are ignored by
the current pipeline and consumed only by future temporal logic / eval):

  corpus entry:
    title, text, idx, timestamp, provenance        # consumed today
    valid_from, valid_to                            # future: time-scoped retrieval
    claim_id, version, status                       # eval-only gold (conflict/supersede)

  question entry:
    id, question, answer, paragraphs                # consumed by get_gold_docs/answers
    category, question_time, gold_timestamp, gold_claim_id   # temporal eval labels

IMPORTANT: only `title`/`text` are indexed and only `timestamp`/`provenance`
reach the graph, so the gold labels (claim_id/version/status) never leak into
retrieval — the system cannot cheat.

Run:  python reproduce/build_temporal_mvp.py
"""

import os
import json
import random

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
CORPUS_PATH = os.path.join(OUT_DIR, "temporal_mvp_corpus.json")
QUERY_PATH = os.path.join(OUT_DIR, "temporal_mvp.json")


# --- Versioned claims: each has an ordered list of versions (old -> new). ----
# A version with valid_to=None is the currently-active one.
VERSIONED_CLAIMS = [
    {"claim_id": "alice_role", "versions": [
        {"value": "Engineer", "timestamp": 2018, "valid_from": 2018, "valid_to": 2021,
         "title": "HR Record 2018 - Alice", "text": "In 2018, Alice worked as an Engineer at the company.",
         "provenance": "HR_Record_2018"},
        {"value": "Manager", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
         "title": "HR Record 2021 - Alice", "text": "As of 2021, Alice works as a Manager at the company.",
         "provenance": "HR_Record_2021"},
    ]},
    {"claim_id": "bob_team", "versions": [
        {"value": "Team Alpha", "timestamp": 2017, "valid_from": 2017, "valid_to": 2020,
         "title": "Org Chart 2017 - Bob", "text": "In 2017, Bob was a member of Team Alpha.",
         "provenance": "Org_Chart_2017"},
        {"value": "Team Beta", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
         "title": "Org Chart 2020 - Bob", "text": "Since 2020, Bob is a member of Team Beta.",
         "provenance": "Org_Chart_2020"},
    ]},
    {"claim_id": "hq_city", "versions": [
        {"value": "Boston", "timestamp": 2015, "valid_from": 2015, "valid_to": 2020,
         "title": "Facilities Report 2015", "text": "In 2015, the company headquarters was located in Boston.",
         "provenance": "Facilities_2015"},
        {"value": "Denver", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
         "title": "Facilities Report 2020", "text": "Since 2020, the company headquarters is located in Denver.",
         "provenance": "Facilities_2020"},
    ]},
    {"claim_id": "projectx_owner", "versions": [
        {"value": "Alice", "timestamp": 2019, "valid_from": 2019, "valid_to": 2021,
         "title": "Project X Charter 2019", "text": "In 2019, Project X was owned by Alice.",
         "provenance": "ProjectX_Charter_2019"},
        {"value": "Eve", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
         "title": "Project X Charter 2021", "text": "As of 2021, Project X is owned by Eve.",
         "provenance": "ProjectX_Charter_2021"},
    ]},
    {"claim_id": "vacation_days", "versions": [
        {"value": "15", "timestamp": 2016, "valid_from": 2016, "valid_to": 2021,
         "title": "Leave Policy 2016", "text": "In 2016, employees received 15 vacation days per year.",
         "provenance": "Leave_Policy_2016"},
        {"value": "20", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
         "title": "Leave Policy 2021", "text": "Since 2021, employees receive 20 vacation days per year.",
         "provenance": "Leave_Policy_2021"},
    ]},
    {"claim_id": "grace_title", "versions": [
        {"value": "Analyst", "timestamp": 2017, "valid_from": 2017, "valid_to": 2020,
         "title": "HR Record 2017 - Grace", "text": "In 2017, Grace held the title of Analyst.",
         "provenance": "HR_Record_2017"},
        {"value": "Lead Analyst", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
         "title": "HR Record 2020 - Grace", "text": "Since 2020, Grace holds the title of Lead Analyst.",
         "provenance": "HR_Record_2020"},
    ]},
    {"claim_id": "flagship", "versions": [
        {"value": "ProductA", "timestamp": 2016, "valid_from": 2016, "valid_to": 2021,
         "title": "Product Catalog 2016", "text": "In 2016, the company's flagship product was ProductA.",
         "provenance": "Catalog_2016"},
        {"value": "ProductB", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
         "title": "Product Catalog 2021", "text": "Since 2021, the company's flagship product is ProductB.",
         "provenance": "Catalog_2021"},
    ]},
    {"claim_id": "bob_manager", "versions": [
        {"value": "Carol", "timestamp": 2017, "valid_from": 2017, "valid_to": 2020,
         "title": "Reporting Lines 2017", "text": "In 2017, Bob's manager was Carol.",
         "provenance": "Reporting_2017"},
        {"value": "David", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
         "title": "Reporting Lines 2020", "text": "Since 2020, Bob's manager is David.",
         "provenance": "Reporting_2020"},
    ]},
    {"claim_id": "office_count", "versions": [
        {"value": "3", "timestamp": 2015, "valid_from": 2015, "valid_to": 2019,
         "title": "Annual Report 2015", "text": "In 2015, the company operated 3 offices.",
         "provenance": "Annual_Report_2015"},
        {"value": "5", "timestamp": 2019, "valid_from": 2019, "valid_to": None,
         "title": "Annual Report 2019", "text": "Since 2019, the company operates 5 offices.",
         "provenance": "Annual_Report_2019"},
    ]},
    {"claim_id": "ceo", "versions": [
        {"value": "Helen", "timestamp": 2012, "valid_from": 2012, "valid_to": 2020,
         "title": "Leadership 2012", "text": "In 2012, the company's CEO was Helen.",
         "provenance": "Leadership_2012"},
        {"value": "Ian", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
         "title": "Leadership 2020", "text": "Since 2020, the company's CEO is Ian.",
         "provenance": "Leadership_2020"},
    ]},
]

# --- Static facts: single version, never change (used for control + as the ---
# --- second hop of multi-hop questions). ------------------------------------
STATIC_CLAIMS = [
    {"claim_id": "carol_role", "value": "CTO", "timestamp": 2010, "valid_from": 2010, "valid_to": None,
     "title": "HR Record - Carol", "text": "Carol is the Chief Technology Officer (CTO) of the company.",
     "provenance": "HR_Record_Carol"},
    {"claim_id": "company_founded", "value": "2008", "timestamp": 2008, "valid_from": 2008, "valid_to": None,
     "title": "Company History", "text": "The company was founded in 2008.",
     "provenance": "Company_History"},
    {"claim_id": "eve_role", "value": "Director", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
     "title": "HR Record - Eve", "text": "Eve is a Director at the company.",
     "provenance": "HR_Record_Eve"},
    {"claim_id": "beta_lead", "value": "Frank", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
     "title": "Team Beta Overview", "text": "Team Beta is led by Frank.",
     "provenance": "Team_Beta_Overview"},
    {"claim_id": "denver_country", "value": "USA", "timestamp": 2015, "valid_from": 2015, "valid_to": None,
     "title": "Geography Reference", "text": "Denver is a city located in the USA.",
     "provenance": "Geography_Reference"},
    {"claim_id": "ian_prev", "value": "TechCorp", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
     "title": "Executive Bio - Ian", "text": "Before joining the company, Ian previously worked at TechCorp.",
     "provenance": "Executive_Bio_Ian"},
    {"claim_id": "productb_category", "value": "cloud platform", "timestamp": 2021, "valid_from": 2021, "valid_to": None,
     "title": "ProductB Spec", "text": "ProductB is a cloud platform.",
     "provenance": "ProductB_Spec"},
    {"claim_id": "david_office", "value": "Denver", "timestamp": 2020, "valid_from": 2020, "valid_to": None,
     "title": "Staff Directory - David", "text": "David works at the Denver office.",
     "provenance": "Staff_Directory_David"},
]

# --- Distractors: unrelated facts so retrieval is non-trivial. --------------
DISTRACTORS = [
    {"claim_id": "distractor_karen", "value": None, "timestamp": 2019, "valid_from": 2019, "valid_to": None,
     "title": "HR Record - Karen", "text": "Karen is a Designer in the Marketing team.",
     "provenance": "HR_Record_Karen"},
    {"claim_id": "distractor_cafeteria", "value": None, "timestamp": 2017, "valid_from": 2017, "valid_to": None,
     "title": "Facilities Note", "text": "The company cafeteria opened in 2017.",
     "provenance": "Facilities_Note"},
    {"claim_id": "distractor_gamma", "value": None, "timestamp": 2018, "valid_from": 2018, "valid_to": None,
     "title": "Team Gamma Overview", "text": "Team Gamma focuses on research.",
     "provenance": "Team_Gamma_Overview"},
    {"claim_id": "distractor_leo", "value": None, "timestamp": 2020, "valid_from": 2020, "valid_to": None,
     "title": "HR Record - Leo", "text": "Leo is a Data Scientist.",
     "provenance": "HR_Record_Leo"},
    {"claim_id": "distractor_retreat", "value": None, "timestamp": 2016, "valid_from": 2016, "valid_to": None,
     "title": "Events Calendar", "text": "The annual company retreat is held every summer.",
     "provenance": "Events_Calendar"},
]


def build_corpus():
    """Returns (corpus list, registry) where registry maps (claim_id, version) -> passage."""
    corpus = []
    registry = {}

    def add(passage, claim_id, version, valid_to):
        passage = dict(passage)
        passage["claim_id"] = claim_id
        passage["version"] = version
        passage["status"] = "active" if valid_to is None else "superseded"
        corpus.append(passage)
        registry[(claim_id, version)] = passage

    for claim in VERSIONED_CLAIMS:
        for i, v in enumerate(claim["versions"], start=1):
            add({"title": v["title"], "text": v["text"], "timestamp": v["timestamp"],
                 "valid_from": v["valid_from"], "valid_to": v["valid_to"], "provenance": v["provenance"]},
                claim["claim_id"], i, v["valid_to"])

    for c in STATIC_CLAIMS + DISTRACTORS:
        add({"title": c["title"], "text": c["text"], "timestamp": c["timestamp"],
             "valid_from": c["valid_from"], "valid_to": c["valid_to"], "provenance": c["provenance"]},
            c["claim_id"], 1, c["valid_to"])

    # Shuffle so versions of the same claim aren't trivially adjacent, then idx.
    rng = random.Random(SEED)
    rng.shuffle(corpus)
    for idx, passage in enumerate(corpus):
        passage["idx"] = idx

    return corpus, registry


# --- Question specs: (category, question, answer, question_time, gold_claim_id, ---
# --- gold_version, [supporting (claim_id, version) ...]). The first supporting ---
# --- entry's version is the "gold version"; others may be marked non-supporting. -
def build_questions(registry):
    # (category, qid, question, answer, q_time, gold_claim_id, gold_version,
    #  supporting=[(claim_id, version, is_supporting), ...])
    SPECS = [
        # ---- latest_wins (10) : "current/now", gold = newest version ----
        ("latest_wins", "What is Alice's current role?", ["Manager"], 2024, "alice_role", 2,
         [("alice_role", 2, True), ("alice_role", 1, False)]),
        ("latest_wins", "What team is Bob currently on?", ["Team Beta"], 2024, "bob_team", 2,
         [("bob_team", 2, True), ("bob_team", 1, False)]),
        ("latest_wins", "In which city is the company headquarters currently located?", ["Denver"], 2024, "hq_city", 2,
         [("hq_city", 2, True), ("hq_city", 1, False)]),
        ("latest_wins", "Who currently owns Project X?", ["Eve"], 2024, "projectx_owner", 2,
         [("projectx_owner", 2, True), ("projectx_owner", 1, False)]),
        ("latest_wins", "How many vacation days do employees currently receive per year?", ["20"], 2024, "vacation_days", 2,
         [("vacation_days", 2, True), ("vacation_days", 1, False)]),
        ("latest_wins", "What is Grace's current title?", ["Lead Analyst"], 2024, "grace_title", 2,
         [("grace_title", 2, True), ("grace_title", 1, False)]),
        ("latest_wins", "What is the company's current flagship product?", ["ProductB"], 2024, "flagship", 2,
         [("flagship", 2, True), ("flagship", 1, False)]),
        ("latest_wins", "Who is Bob's current manager?", ["David"], 2024, "bob_manager", 2,
         [("bob_manager", 2, True), ("bob_manager", 1, False)]),
        ("latest_wins", "How many offices does the company currently operate?", ["5"], 2024, "office_count", 2,
         [("office_count", 2, True), ("office_count", 1, False)]),
        ("latest_wins", "Who is the company's current CEO?", ["Ian"], 2024, "ceo", 2,
         [("ceo", 2, True), ("ceo", 1, False)]),

        # ---- time_specific (8) : past question_time, gold = old version ----
        ("time_specific", "What was Alice's role in 2019?", ["Engineer"], 2019, "alice_role", 1,
         [("alice_role", 1, True), ("alice_role", 2, False)]),
        ("time_specific", "Which team was Bob on in 2018?", ["Team Alpha"], 2018, "bob_team", 1,
         [("bob_team", 1, True), ("bob_team", 2, False)]),
        ("time_specific", "In which city was the headquarters located in 2017?", ["Boston"], 2017, "hq_city", 1,
         [("hq_city", 1, True), ("hq_city", 2, False)]),
        ("time_specific", "Who owned Project X in 2020?", ["Alice"], 2020, "projectx_owner", 1,
         [("projectx_owner", 1, True), ("projectx_owner", 2, False)]),
        ("time_specific", "How many vacation days did employees receive in 2018?", ["15"], 2018, "vacation_days", 1,
         [("vacation_days", 1, True), ("vacation_days", 2, False)]),
        ("time_specific", "What was Grace's title in 2018?", ["Analyst"], 2018, "grace_title", 1,
         [("grace_title", 1, True), ("grace_title", 2, False)]),
        ("time_specific", "Who was the company's CEO in 2015?", ["Helen"], 2015, "ceo", 1,
         [("ceo", 1, True), ("ceo", 2, False)]),
        ("time_specific", "How many offices did the company have in 2016?", ["3"], 2016, "office_count", 1,
         [("office_count", 1, True), ("office_count", 2, False)]),

        # ---- multihop_updated (6) : 2-hop, the bridge fact was updated ----
        ("multihop_updated", "What is the role of the current owner of Project X?", ["Director"], 2024, "projectx_owner", 2,
         [("projectx_owner", 2, True), ("eve_role", 1, True)]),
        ("multihop_updated", "Who leads the team that Bob is currently on?", ["Frank"], 2024, "bob_team", 2,
         [("bob_team", 2, True), ("beta_lead", 1, True)]),
        ("multihop_updated", "In which country is the company's current headquarters located?", ["USA"], 2024, "hq_city", 2,
         [("hq_city", 2, True), ("denver_country", 1, True)]),
        ("multihop_updated", "Where did the company's current CEO previously work?", ["TechCorp"], 2024, "ceo", 2,
         [("ceo", 2, True), ("ian_prev", 1, True)]),
        ("multihop_updated", "What category does the company's current flagship product belong to?", ["cloud platform"], 2024, "flagship", 2,
         [("flagship", 2, True), ("productb_category", 1, True)]),
        ("multihop_updated", "At which office does Bob's current manager work?", ["Denver"], 2024, "bob_manager", 2,
         [("bob_manager", 2, True), ("david_office", 1, True)]),

        # ---- conflict (6) : time-agnostic phrasing, both versions retrievable, gold = new ----
        ("conflict", "How many vacation days do employees receive?", ["20"], 2024, "vacation_days", 2,
         [("vacation_days", 2, True), ("vacation_days", 1, False)]),
        ("conflict", "What is Alice's job title?", ["Manager"], 2024, "alice_role", 2,
         [("alice_role", 2, True), ("alice_role", 1, False)]),
        ("conflict", "Who is the company's CEO?", ["Ian"], 2024, "ceo", 2,
         [("ceo", 2, True), ("ceo", 1, False)]),
        ("conflict", "What is the company's flagship product?", ["ProductB"], 2024, "flagship", 2,
         [("flagship", 2, True), ("flagship", 1, False)]),
        ("conflict", "Where is the company headquartered?", ["Denver"], 2024, "hq_city", 2,
         [("hq_city", 2, True), ("hq_city", 1, False)]),
        ("conflict", "How many offices does the company have?", ["5"], 2024, "office_count", 2,
         [("office_count", 2, True), ("office_count", 1, False)]),

        # ---- control (6) : static facts, no versioning ----
        ("control", "Who is the company's CTO?", ["Carol"], 2024, "carol_role", 1,
         [("carol_role", 1, True)]),
        ("control", "In which year was the company founded?", ["2008"], 2024, "company_founded", 1,
         [("company_founded", 1, True)]),
        ("control", "What is Eve's role?", ["Director"], 2024, "eve_role", 1,
         [("eve_role", 1, True)]),
        ("control", "Who leads Team Beta?", ["Frank"], 2024, "beta_lead", 1,
         [("beta_lead", 1, True)]),
        ("control", "In which country is Denver located?", ["USA"], 2024, "denver_country", 1,
         [("denver_country", 1, True)]),
        ("control", "What category is ProductB?", ["cloud platform"], 2024, "productb_category", 1,
         [("productb_category", 1, True)]),
    ]

    counters = {}
    questions = []
    for category, question, answer, q_time, gold_claim_id, gold_version, supporting in SPECS:
        counters[category] = counters.get(category, 0) + 1
        paragraphs = []
        for claim_id, version, is_supporting in supporting:
            p = registry[(claim_id, version)]
            paragraphs.append({
                "title": p["title"], "text": p["text"], "idx": p["idx"],
                "is_supporting": is_supporting,
            })
        gold_passage = registry[(gold_claim_id, gold_version)]
        questions.append({
            "id": f"tq/{category}/q{counters[category]}",
            "question": question,
            "answer": answer,
            "paragraphs": paragraphs,
            "category": category,
            "question_time": q_time,
            "gold_timestamp": gold_passage["timestamp"],
            "gold_claim_id": gold_claim_id,
        })
    return questions


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    corpus, registry = build_corpus()
    questions = build_questions(registry)

    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f, indent=2)
    with open(QUERY_PATH, "w") as f:
        json.dump(questions, f, indent=2)

    by_cat = {}
    for q in questions:
        by_cat[q["category"]] = by_cat.get(q["category"], 0) + 1
    print(f"Wrote {len(corpus)} passages -> {CORPUS_PATH}")
    print(f"Wrote {len(questions)} questions -> {QUERY_PATH}")
    print("Questions by category:", by_cat)


if __name__ == "__main__":
    main()
