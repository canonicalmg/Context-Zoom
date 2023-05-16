import openai
from openai.error import RateLimitError
import time
import curses
import json
from transformers import GPT2Tokenizer
import os

OPENAI_API_KEY=os.getenv('OPENAI_API_KEY', '')
openai.api_key = OPENAI_API_KEY  

def gpt_call(prompt, temperature=0.85, max_retries=3):
    messages = [
        {"role": "system", "content": "You are summarizeGPT. Specializing in summarizing text."},
        {"role": "user", "content": prompt}
    ]
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                n=1,
                stop=None,
            )
            message_content = response.choices[0].message.content.strip()
            return message_content
        except RateLimitError as e:
            retries += 1
            # print(f"Rate limit error occurred. Retrying ({retries}/{max_retries})...")
            # Wait for a short duration before retrying
            time.sleep(2)
    # Retry limit reached, return an empty string
    return ""

def summarize_into_blocks(text, num_blocks, block_size):
    # prompt = f"""Given the text below, please summarize it into {num_blocks} blocks of {block_size} tokens each. Each block should be separated by [NEW_BLOCK] This is very important as it will be parsed with python afterwards.
    # TEXT TO SUMMARIZE: 
    # {text}"""

    prompt = f"""
    Please summarize this text into {num_blocks} of {block_size} tokens each. Please output the summaries as a python list of strings like so: ["summary for block 1", "summary for block 2"]. Do not include a trailing comma after the last summary in the list.
    {text}
    """

    response = gpt_call(prompt)
    # print(f"\nRESPONSE from gpt: {response}\n")
    # blocks = []
    # for i in response.split("[NEW_BLOCK]"):
    #     if i:

    #         blocks.append(i.lstrip())
    # return blocks
    try:
        return json.loads(response)
    except Exception as e:
        # print("error in json response", e)
        return []


def chunk_text(text, max_tokens=1000):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
        
    return chunks

def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    return len(tokens)

class TextBlock:
    def __init__(self, text, children=None):
        self.text = text
        self.child_blocks = children if children else []

    def summarize(self):
        num_tokens = count_tokens(self.text)
        # print(f"Summarizing {self.text} of size: {num_tokens}")
        # Stop summarizing if the text is 5 tokens or less
        if num_tokens <= 5:
            return
        # Summarize the text
        summaries = summarize_into_blocks(self.text, 4, num_tokens // 4)
        for summary in summaries:
            summarized_block = TextBlock(summary)

            # Add the new child block to the list of children
            self.child_blocks.append(summarized_block)

        # Recursively summarize each child block
        for child in self.child_blocks:
            child.summarize()

def recursive_summarization(text):
    chunks = chunk_text(text)
    root_blocks = [TextBlock(chunk) for chunk in chunks]
    for root_block in root_blocks:
        root_block.summarize()
    return root_blocks


def get_test_data():
    root_blocks = [
        TextBlock(
            "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
            children=[
                TextBlock("Alice was tired sitting by her sister, with nothing to do.", children=[
                    TextBlock("Alice was tired and bored."),
                    TextBlock("She had nothing to do."),
                ]),
                TextBlock("She had looked at her sister's book, but found it uninteresting.", children=[
                    TextBlock("She peeped into her sister's book."),
                    TextBlock("The book had no pictures or conversations."),
                ]),
            ]
        ),
        TextBlock(
            "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
            children=[
                TextBlock("Alice was considering making a daisy-chain, but was feeling too hot and sleepy.", children=[
                    TextBlock("It was a hot day."),
                    TextBlock("She was contemplating the effort of making a daisy-chain."),
                ]),
                TextBlock("Suddenly, a White Rabbit ran close by her.", children=[
                    TextBlock("A White Rabbit appeared suddenly."),
                    TextBlock("The rabbit ran close to Alice."),
                ]),
            ]
        ),
        TextBlock(
            "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.",
            children=[
                TextBlock("Alice didn't find it particularly strange when the Rabbit spoke and expressed concern about being late. Though she later realized she should have been more surprised, at the time it felt quite natural. The Rabbit then hurriedly took out a watch from its waistcoat-pocket.", children=[
                    TextBlock("Alice found it normal when the Rabbit expressed worry about being late.", children=[
                        TextBlock("Alice wasn't surprised by the Rabbit's worry."),
                        TextBlock("The Rabbit checked its waistcoat-pocket watch in haste.")
                    ]),
                    TextBlock("The Rabbit, in a hurry, checked a watch from its waistcoat-pocket.", children=[
                        TextBlock("Alice was intrigued by a uniquely accessorized rabbit."),
                        TextBlock("The rabbit's rush prompts Alice's pursuit.")
                    ]),
                ]),
                TextBlock("Seeing the Rabbit with a watch and a waistcoat-pocket sparked Alice's curiosity. She had never seen such a sight before. Compelled by her curiosity, Alice chased after the Rabbit across the field, and was fortunate enough to see it disappear down a large rabbit-hole under the hedge.", children=[
                    TextBlock("A rabbit with a waistcoat-pocket and a watch intrigued Alice, a sight she had never seen before", children=[
                        TextBlock("Alice was captivated by a rabbit with human accessories."),
                        TextBlock("Alice had never before seen such a rabbit.")
                    ]),
                    TextBlock("Driven by curiosity, Alice ran after the rabbit, witnessing it vanish down a large rabbit-hole under a hedge.", children=[
                        TextBlock("Alice's curiosity led her to chase the rabbit."),
                        TextBlock("She saw the rabbit disappear down a large hole.")
                    ]),
                ]),
            ]
        ),
        # Add more root blocks as needed...
    ]
    return root_blocks

def main(stdscr):
    # Turn off cursor blinking
    curses.curs_set(0)

    # Color setup
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.
    """.lstrip()
    # root_blocks = recursive_summarization(text)
    root_blocks = get_test_data()

    current_blocks = [root_blocks]
    current_indices = [0]

    column_width = 45  # adjust according to your needs
    space_between_columns = 3  # adjust according to your needs

    while True:
        stdscr.clear()

        # Draw the summaries of the children of the current block
        for col, blocks in enumerate(current_blocks):
            line_no = 0
            for row, block in enumerate(blocks):
                if row == current_indices[col]:
                    stdscr.attron(curses.color_pair(1))
                for i in range(0, len(block.text), column_width):
                    stdscr.addstr(line_no, col * (column_width + space_between_columns), block.text[i:i+column_width])
                    line_no += 1
                line_no += 1  # add a blank line between blocks
                stdscr.attroff(curses.color_pair(1))

        # Wait for next input
        k = stdscr.getch()

        # Navigate between blocks with up and down keys
        if k == curses.KEY_UP:
            current_indices[-1] = max(0, current_indices[-1] - 1)
        elif k == curses.KEY_DOWN:
            current_indices[-1] = min(len(current_blocks[-1]) - 1, current_indices[-1] + 1)
        elif k == curses.KEY_RIGHT:
            if current_blocks[-1][current_indices[-1]].child_blocks:
                current_blocks.append(current_blocks[-1][current_indices[-1]].child_blocks)
                current_indices.append(0)
        elif k == curses.KEY_LEFT:
            if len(current_blocks) > 1:
                current_blocks.pop()
                current_indices.pop()

if __name__ == "__main__":
    curses.wrapper(main)

