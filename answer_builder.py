def build_human_answer(results):
    """
    Converts raw retrieved text into readable English
    """

    # Extract only text from (text, score) tuples
    texts = [item[0] for item in results]

    text = " ".join(texts)

    # remove duplicates & excessive spaces
    text = " ".join(text.split())

    # limit size
    if len(text) > 600:
        text = text[:600] + "..."

    return (
        "Based on the uploaded document, the conclusion is:\n\n"
        + text
    )