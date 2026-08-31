import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAI API key exists and begins with {openai_api_key[:8]}")
else:
    print("OPENAI_API_KEY is not set")
    # raise ValueError("OPENAI_API_KEY is not set")
    
openai = OpenAI()

MODEL = "gpt-4.1-mini"

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."


def chatbot(message, history):
    relevant_system_message = system_message
    # print(history)
    new_history = [{"role":h["role"], "content":h["content"]} for h in history]
    # print("=====")
    # print(new_history)
    # print("=====")
    
    if "belt" in message.lower()  or "hats" in message.lower():
        relevant_system_message += " The store does not sell belts or hats; if you are asked for those, be sure to point out other items on sale."

    # initiate the chat model:
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": relevant_system_message},
            *history,
            {"role": "user", "content": message}
        ],
        stream=True
    )
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response
        
    # return chat.choices[0].message.content
    

demo = gr.ChatInterface(
    fn=chatbot,
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
    #demo.launch(share=True)