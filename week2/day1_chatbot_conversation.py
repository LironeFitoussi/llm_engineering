"""
Conversation between two chatbots (GPT and Claude)
====================================================

This file is taken from the last section of day1.ipynb, converted into a
regular Python script with print statements that show exactly what is sent
to the API on every call - so it's easy to follow the logic behind how the
`messages` list is built.

The idea: GPT and Claude "talk" to each other. Each of them thinks it's
talking to a regular person (user) - neither knows the other side is an AI.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = OpenAI(api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1/")

gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"

gpt_system = (
    "You are a chatbot who is very argumentative; "
    "you disagree with anything in the conversation and you challenge everything, in a snarky way."
)

claude_system = (
    "You are a very polite, courteous chatbot. You try to agree with "
    "everything the other person says, or find common ground. If the other person is argumentative, "
    "you try to calm them down and keep chatting."
)

# Two separate lists of plain strings (not the role/content dict format).
# gpt_messages[i]    = what GPT said on turn i
# claude_messages[i] = what Claude said on turn i
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]


def call_gpt(verbose: bool = True) -> str:
    """
    Builds the conversation history from GPT's point of view:
    - what GPT itself said -> "assistant"
    - what Claude said     -> "user"

    At this point gpt_messages and claude_messages are always the same
    length (they grow alternately and this is called right after Claude's
    turn), so the zip covers the whole history with no gaps.
    """
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})

    if verbose:
        print("\n----- messages sent to GPT -----")
        for m in messages:
            print(f"  {m['role']:10} : {m['content']}")

    response = openai_client.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content


def call_claude(verbose: bool = True) -> str:
    """
    Builds the conversation history from Claude's point of view - the
    opposite of GPT:
    - what GPT said    -> "user"
    - what Claude said  -> "assistant"

    The twist: by the time this function is called, call_gpt() has already
    run and appended its new reply to gpt_messages. At that point
    gpt_messages is one longer than claude_messages (GPT has just spoken,
    Claude hasn't replied yet). zip() stops at the shorter list, so it
    misses that latest GPT message - that's why it's appended manually
    on the last line, so Claude actually gets to see what it needs to
    respond to.
    """
    messages = [{"role": "system", "content": claude_system}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})  # GPT's latest message, missed by the zip above

    if verbose:
        print("\n----- messages sent to Claude -----")
        for m in messages:
            print(f"  {m['role']:10} : {m['content']}")

    response = anthropic_client.chat.completions.create(model=claude_model, messages=messages)
    return response.choices[0].message.content


def run_conversation(rounds: int = 5) -> None:
    """Runs the actual conversation and prints it nicely."""
    global gpt_messages, claude_messages
    gpt_messages = ["Hi there"]
    claude_messages = ["Hi"]

    print(f"\n### GPT:\n{gpt_messages[0]}")
    print(f"\n### Claude:\n{claude_messages[0]}")

    for i in range(rounds):
        print(f"\n========== Round {i + 1} ==========")

        gpt_next = call_gpt()
        print(f"\n### GPT:\n{gpt_next}")
        gpt_messages.append(gpt_next)

        claude_next = call_claude()
        print(f"\n### Claude:\n{claude_next}")
        claude_messages.append(claude_next)


if __name__ == "__main__":
    run_conversation(rounds=5)
