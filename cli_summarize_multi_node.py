python
def recursive_summarization(text):
    """
    Recursively summarizes the given text by chunking it into smaller blocks and summarizing each block.

    Args:
        text (str): The input text to be summarized.

    Returns:
        list: A list of summarized text blocks.
    """
    # Create the root text block
    root = TextBlock(text)

    # Chunk the text into smaller blocks
    chunks = chunk_text(text)

    # Create a TextBlock for each chunk and add it as a child block of the root
    for chunk in chunks:
        root.child_blocks.append(TextBlock(chunk))

    # Summarize each child block
    for child in root.child_blocks:
        child.summarize()

    # Return the list of summarized child blocks
    return root.child_blocks
