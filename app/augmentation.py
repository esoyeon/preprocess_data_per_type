import nltk
from nltk.corpus import wordnet
import random

# Download required NLTK data
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


def get_synonyms(word: str) -> list:
    """Get synonyms for a word using WordNet"""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(set(synonyms))


def synonym_replacement(text: str, n_changes: int = 3) -> str:
    """Replace n words in the text with their synonyms"""
    words = text.split()
    if not words:
        return text

    n_changes = min(n_changes, len(words))

    for _ in range(n_changes):
        replace_idx = random.randint(0, len(words) - 1)
        word = words[replace_idx]
        synonyms = get_synonyms(word)

        if synonyms:
            words[replace_idx] = random.choice(synonyms)

    return " ".join(words)


def random_swap(text: str, n_swaps: int = 3) -> str:
    """Randomly swap n pairs of words in the text"""
    words = text.split()
    if len(words) < 2:
        return text

    n_swaps = min(n_swaps, len(words) // 2)

    for _ in range(n_swaps):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]

    return " ".join(words)


def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words from the text with probability p"""
    words = text.split()
    if len(words) == 1:
        return text

    words = [word for word in words if random.random() > p]

    if not words:
        return text

    return " ".join(words)


def augment_text(text: str) -> str:
    """
    Augment the text using three techniques:
    1. Synonym Replacement
    2. Random Swap
    3. Random Deletion
    """
    # Apply augmentation techniques sequentially
    augmented = synonym_replacement(text)
    augmented = random_swap(augmented)
    augmented = random_deletion(augmented)

    return augmented
