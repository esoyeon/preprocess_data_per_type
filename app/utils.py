def validate_text(text: str) -> bool:
    """
    Validate if the input text is proper and not empty
    """
    if not text or not isinstance(text, str):
        return False
    return True

def clean_text(text: str) -> str:
    """
    Basic text cleaning
    """
    return text.strip() 