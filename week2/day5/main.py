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
"""

get_ticket_price("Paris")
def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities

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
    response = openai.chat.completions.create(model=MODEL,messages=messages,tools=tools)
    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        tool_message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(tool_message)
        messages.append(tool_message)
        messages.extend(responses)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )

    reply = response.choices[0].message.content

    if cities:
        image = artist(cities[0])

    # ChatInterface takes the reply itself; extras go to additional_outputs
    return reply, image, speak(reply)


# IMAGE GENERATION
# city = "New York City"
# print(f"Generating image for {city}...")
# image = artist(city)
# image.save("new_york_city.png")
# print("Saved to new_york_city.png")

image_output = gr.Image(label="Destination", height=400)
audio_output = gr.Audio(label="Voice", autoplay=True)

demo = gr.ChatInterface(
    fn=chatbot,
    type="messages",
    additional_outputs=[image_output, audio_output],
)

if __name__ == "__main__":
    demo.launch()