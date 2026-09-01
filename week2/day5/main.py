import json
import os
import tempfile
import uuid
import gradio as gr
from tools import get_ticket_price, tools
from agents import artist, talker, openai

# MODEL = "gpt-oss:20b "
MODEL = "gpt-4.1-mini"

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
You can look up ticket prices and show the customer a picture of a destination.
Decide for yourself which of those the conversation calls for.
"""

def handle_tool_calls(message):
    """Run whichever tools the model decided to call.

    Returns the tool results to feed back to the model, plus an image if the
    model chose to call the artist.
    """
    responses = []
    image = None
    for tool_call in message.tool_calls:
        name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if name == "get_ticket_price":
            content = get_ticket_price(arguments.get("destination_city"))
        elif name == "generate_destination_image":
            city = arguments.get("city")
            image = artist(city)
            content = f"An image of {city} is now displayed to the customer."
        else:
            content = f"Unknown tool: {name}"

        responses.append({
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        })
    return responses, image

def speak(reply):
    """talker() returns raw mp3 bytes; gr.Audio needs a file path."""
    audio_bytes = talker(reply)
    path = os.path.join(tempfile.gettempdir(), f"flightai_{uuid.uuid4().hex}.mp3")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path

def chatbot(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        tool_message = response.choices[0].message
        responses, new_image = handle_tool_calls(tool_message)
        image = new_image or image
        messages.append(tool_message)
        messages.extend(responses)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

    reply = response.choices[0].message.content

    # ChatInterface takes the reply itself; extras go to additional_outputs
    return reply, image, speak(reply)


# IMAGE GENERATION
# city = "New York City"
# print(f"Generating image for {city}...")
# image = artist(city)
# image.save("new_york_city.png")
# print("Saved to new_york_city.png")

# Created unrendered, then .render()ed inside the Blocks below so they live in
# the same Blocks scope as the ChatInterface -- otherwise they never show up.
image_output = gr.Image(label="Destination", height=400)
audio_output = gr.Audio(label="Voice", autoplay=True)

with gr.Blocks(title="FlightAI") as demo:
    with gr.Row():
        with gr.Column(scale=3):
            gr.ChatInterface(
                fn=chatbot,
                type="messages",
                additional_outputs=[image_output, audio_output],
            )
        with gr.Column(scale=2):
            image_output.render()
            audio_output.render()

if __name__ == "__main__":
    demo.launch()