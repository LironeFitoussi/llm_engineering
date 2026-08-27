import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

openai = OpenAI(api_key=openai_api_key)
anthropic = OpenAI(api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1")

system_message = "You are a funny assistant"
def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result
    
def stream_claude(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = anthropic.chat.completions.create(
        model='claude-sonnet-4-5-20250929',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result
        
def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

message_input = gr.Textbox(label="Your message:", info="Enter a message for the LLM", lines=7)
model_selector = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_model,
    title="LLMs", 
    inputs=[message_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["Explain the Transformer architecture to a layperson", "GPT"],
            ["Explain the Transformer architecture to an aspiring AI engineer", "Claude"]
        ], 
    flagging_mode="never"
    )
view.launch()