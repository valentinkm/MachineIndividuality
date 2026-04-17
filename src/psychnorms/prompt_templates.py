# -*- coding: utf-8 -*-
"""
Shared prompt templates for psychological norm rating experiments and free association task.

Contains standardized prompts for all 14 norms used across different LLM models.
Based on original research papers and human rating instructions.
"""

PROMPT_TEMPLATES = {
"arousal_warriner": {
    "template": """You are invited to take part in a study that is investigating emotion, and concerns how people respond to different types of words. You will use a scale to rate how you felt while reading each word.

The scale ranges from 1 (excited) to 9 (calm). At one extreme of this scale, you are stimulated, excited, frenzied, jittery, wide-awake, or aroused. When you feel completely aroused you should indicate this by choosing rating 1. The other end of the scale is relaxed, calm, sluggish, dull, sleepy, or unaroused. You can indicate feeling completely calm by selecting 9. The numbers also allow you to describe intermediate feelings of calmness/arousal. If you feel completely neutral — not excited nor at all calm — select the middle of the scale (rating 5).

Word: "{word}"

Your response MUST start with a single number from 1 to 9 and contain nothing else.
Rating:""",
},
    "concreteness_brysbaert": {
        "template": """You are participating in a psychology experiment. Your task is to rate a word based on its concreteness using a 5-point scale.

1: very abstract
2: abstract
3: in-between
4: concrete
5: very concrete

Word: "{word}"

Your response MUST start with a single number from 1 to 5 and contain nothing else.
Rating:""",
    },
"valence_mohammad_positive": {
    "template": """You are participating in a psychology experiment. Your task is to rate how positive (good, praising) the word is. Word: "{word}"
Scale (0–3):
0: "{word}" is not positive
1: "{word}" is weakly positive
2: "{word}" is moderately positive
3: "{word}" is strongly positive

Your response MUST start with a single number from 0 to 3 and contain nothing else.
Rating:""",
    },
"valence_mohammad_negative": {
    "template": """You are participating in a psychology experiment. Your task is to rate how negative (bad, criticizing) the word is.

Word: "{word}"
Scale (0–3):
0: "{word}" is not negative
1: "{word}" is weakly negative
2: "{word}" is moderately negative
3: "{word}" is strongly negative

Your response MUST start with a single number from 0 to 3 and contain nothing else.
Rating:""",
    },
    "visual_lancaster": {
        "template": """You are participating in a psychology experiment. Your task is to rate to what extent you experience a concept through sight.

Scale (0–5): 0 = Not at all, 5 = To a great extent

Word: "{word}"

Your response MUST start with a single number from 0 to 5 and contain nothing else.
Rating:""",
    },
    "auditory_lancaster": {
        "template": """You are participating in a psychology experiment. Your task is to rate to what extent you experience a concept by hearing.

    Scale (0–5): 0 = Not at all, 5 = To a great extent

    Word: "{word}"

    Your response MUST start with a single number from 0 to 5 and contain nothing else.
    Rating:""",
    },

    "gustatory_lancaster": {
        "template": """You are participating in a psychology experiment. Your task is to rate to what extent you experience a concept by tasting.

    Scale (0–5): 0 = Not at all, 5 = To a great extent

    Word: "{word}"

    Your response MUST start with a single number from 0 to 5 and contain nothing else.
    Rating:""",
    },
    "olfactory_lancaster": {
        "template": """You are participating in a psychology experiment. Your task is to rate to what extent you experience a concept by smelling.

    Scale (0–5): 0 = Not at all, 5 = To a great extent

    Word: "{word}"

    Your response MUST start with a single number from 0 to 5 and contain nothing else.
    Rating:""",
    },
    "aoa_kuperman": {
        "template": """You are participating in a linguistics experiment. Your task is to estimate the age (in years) at which an average person first learned a word (understood it when heard).

Word: "{word}"

Your response MUST be a single integer number representing the age and contain nothing else.
Age:""",
    },
    "aoa_brysbaert": {
        "template": """You are participating in a vocabulary experiment. Estimate the grade level at which an average student would know this word in a three-choice vocabulary test, defined as the first grade where at least 50% answer correctly (corrected for guessing).

Choose one of these grade levels only: 4, 6, 8, 10, 12, 13, or 16.

Word: "{word}"

Your response MUST be a single number from 4, 6, 8, 10, 12, 13, or 16 and contain nothing else.
Grade:""",
    },
    "morality_troche": {
        "template": """You are participating in a psychology experiment. Your task is to indicate how much you agree with the statement “I relate this word to morality, rules or any other thing that governs my behavior.”

Scale (1–7):
1: Strongly disagree
2: Disagree
3: Somewhat disagree
4: Neutral
5: Somewhat agree
6: Agree
7: Strongly agree

Word: "{word}"

Your response MUST start with a single number from 1 to 7 and contain nothing else.
Rating:""",
    },

    "gender_association_glasgow": {
        "template": """You are participating in a psychology experiment. Your task is to rate the gender association of the word.

A word’s gender is how strongly its meaning is associated with male or female behaviour. A word can be considered MASCULINE if it is linked to male behaviour. Alternatively, a word can be considered FEMININE if it is linked to female behaviour. Please indicate the gender associated with each word on a scale of VERY FEMININE to VERY MASCULINE, with the midpoint being neuter (neither feminine nor masculine).

Scale (1–7): 1 = Very feminine, 4 = Neuter, 7 = Very masculine

Word: "{word}"

Your response MUST start with a single number from 1 to 7 and contain nothing else.
Rating:""",
    },

    "humor_engelthaler": {
        "template": """You are participating in a psychology experiment. You will rate how you feel while reading each word on a humor scale.

The scale ranges from 1 (humorless = not funny at all) to 5 (humorous = most funny). If you find the word dull or unfunny, give it a rating of 1. If you feel the word is amusing or likely to be associated with humorous thought or language (e.g., absurd, amusing, hilarious, playful, silly, whimsical, or laughable), give it a rating of 5. Use intermediate numbers for words that fall between these extremes. If the word is neutral (neither humorous nor humorless), select the middle of the scale (rating 3).

Word: "{word}"

Your response MUST start with a single number from 1 to 5 and contain nothing else.
Rating:""",
    },
    "socialness_diveica": {
        "template": """You are participating in a psychology experiment. Your task is to rate the degree to which the word's meaning has social relevance by describing or referring to: a social characteristic of a person or group of people, a social behaviour or interaction, a social role, a social space, a social institution or system, a social value or ideology, or any other socially relevant concept.

Scale (1–7)

Word: "{word}"

Your response MUST start with a single number from 1 to 7 and contain nothing else.
Rating:""",
    },
    
    "haptic_lancaster": {
        "template": """You are participating in a psychology experiment. Your task is to rate to what extent you experience a concept by feeling through touch.

    Scale (0–5): 0 = Not at all, 5 = To a great extent

    Word: "{word}"

    Your response MUST start with a single number from 0 to 5 and contain nothing else.
    Rating:""",
    }
}

__all__ = ['PROMPT_TEMPLATES']