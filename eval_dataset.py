"""
Labeled retrieval evaluation set.

Each entry pairs a natural-language question with the ground-truth
(source_file, page_number) of the chunk(s) that should be retrieved to
answer it correctly. Ground truth was determined by manually reading each
document section (see scripts/eval_retrieval.py docstring for the
methodology) -- not guessed or auto-generated, so the numbers this
produces are meaningful rather than circular.

Used by scripts/eval_retrieval.py to compute Hit@K, MRR, and citation
precision -- standard IR metrics that let retrieval quality be reported
as a number instead of "the demo looked fine."

Includes two genuinely out-of-scope questions (expected_sources=None) to
verify the grounding guardrail's true-negative rate, not just its
true-positive rate.
"""

EVAL_SET = [
    {
        "query": "How many microorganisms live in the human gut?",
        "expected_sources": [("Gut_Microbiome_and_Human_Health.pdf", 1)],
    },
    {
        "query": "What is the gut-brain axis?",
        "expected_sources": [("Gut_Microbiome_and_Human_Health.pdf", 2)],
    },
    {
        "query": "How does antibiotic use affect the gut microbiome?",
        "expected_sources": [("Gut_Microbiome_and_Human_Health.pdf", 3)],
    },
    {
        "query": "When and where were hydrothermal vents discovered?",
        "expected_sources": [("Hydrothermal_Vents_Deep_Ocean.docx", 2)],
    },
    {
        "query": "What is chemosynthesis and how does it differ from photosynthesis?",
        "expected_sources": [("Hydrothermal_Vents_Deep_Ocean.docx", 3)],
    },
    {
        "query": "Why are hydrothermal vents relevant to astrobiology and the origin of life?",
        "expected_sources": [("Hydrothermal_Vents_Deep_Ocean.docx", 5)],
    },
    {
        "query": "How did Johannes Gutenberg's printing innovation differ from earlier printing methods?",
        "expected_sources": [("Printing_Press_Information_Revolution.docx", 3)],
    },
    {
        "query": "How did the printing press contribute to the Protestant Reformation?",
        "expected_sources": [("Printing_Press_Information_Revolution.docx", 5)],
    },
    {
        "query": "How did printing standardize European languages?",
        "expected_sources": [("Printing_Press_Information_Revolution.docx", 6)],
    },
    {
        "query": "What causes the intermittency problem in solar and wind energy?",
        "expected_sources": [("Renewable_Energy_Transition_Report.pdf", 2)],
    },
    {
        "query": "What policy obstacles slow down renewable energy deployment?",
        "expected_sources": [("Renewable_Energy_Transition_Report.pdf", 3)],
    },
    {
        "query": "Why do banks typically not fund early-stage startups?",
        "expected_sources": [("Startup_Fundraising_VC_Guide.txt", 2)],
    },
    {
        "query": "What is the typical size of a pre-seed funding round?",
        "expected_sources": [("Startup_Fundraising_VC_Guide.txt", 3)],
    },
    {
        "query": "What do venture investors look for when evaluating a startup pitch?",
        "expected_sources": [("Startup_Fundraising_VC_Guide.txt", 5)],
    },
    {
        "query": "What fundraising mistakes do founders commonly make?",
        "expected_sources": [("Startup_Fundraising_VC_Guide.txt", 6)],
    },
    # --- Out-of-scope questions: expect the guardrail to decline, not guess ---
    {
        "query": "What is the capital of France?",
        "expected_sources": None,
    },
    {
        "query": "Who won the 2024 Super Bowl?",
        "expected_sources": None,
    },
]
