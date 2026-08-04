# fmt: off
# Fifteen demonstrations, matching the shot count Mallen et al. (2023) use for
# open models: "we use 15-shot prompting for all GPT-neo and OPT models".
#
# The paper does not publish its own fifteen, so these are written here. Two
# things make them faithful in shape even though the content differs:
#
#   - Every question uses a relation template taken verbatim from the dataset,
#     so the surface forms match what the model will be scored on. PopQA phrases
#     these precisely -- "Who *was* the director of X?" but "Who *is* the author
#     of X?" -- and guessing them wrong would make the demonstrations teach a
#     format the test questions never use.
#   - Answers are bare entities in the dataset's own style, including its
#     preference for "association football" over "soccer".
#
# Subjects are high-popularity on purpose: they sit in the head of the
# distribution PopQA is built to contrast against its long tail. A handful may
# also appear among the 14,267 scored questions, which is the same contamination
# the paper's own in-dataset demonstrations would carry, and bounded at roughly
# 0.1% of the split.
POPQA_FIXED_FEWSHOT = [
    # "What is [subj]'s occupation?"
    {"question": "What is Albert Einstein's occupation?", "answer": ["physicist"]},
    {"question": "What is Frida Kahlo's occupation?", "answer": ["painter"]},
    {"question": "What is Ludwig van Beethoven's occupation?", "answer": ["composer"]},
    # "Who is the author of [subj]?"
    {"question": "Who is the author of Pride and Prejudice?", "answer": ["Jane Austen"]},
    {"question": "Who is the author of Nineteen Eighty-Four?", "answer": ["George Orwell"]},
    # "Who was the director of [subj]?"
    {"question": "Who was the director of Jurassic Park?", "answer": ["Steven Spielberg"]},
    {"question": "Who was the director of Pulp Fiction?", "answer": ["Quentin Tarantino"]},
    # "Who was the screenwriter for [subj]?"
    {"question": "Who was the screenwriter for Chinatown?", "answer": ["Robert Towne"]},
    {"question": "Who was the screenwriter for Alien?", "answer": ["Dan O'Bannon"]},
    # "Who was the producer of [subj]?"
    {"question": "Who was the producer of Thriller?", "answer": ["Quincy Jones"]},
    {"question": "Who was the producer of Nevermind?", "answer": ["Butch Vig"]},
    # "What genre is [subj]?"
    {"question": "What genre is Abbey Road?", "answer": ["rock"]},
    {"question": "What genre is The Godfather?", "answer": ["crime film"]},
    # "Who is the father of [subj]?"
    {"question": "Who is the father of Alexander the Great?", "answer": ["Philip II of Macedon"]},
    # "What sport does [subj] play?"
    {"question": "What sport does Lionel Messi play?", "answer": ["association football"]},
]
