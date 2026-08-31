import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import json
from tools.database import *

load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAI API key exists and begins with {openai_api_key[:8]}")
else:
    print("OPENAI_API_KEY is not set")
    # raise ValueError("OPENAI_API_KEY is not set")
    
openai = OpenAI()

MODEL = "gpt-4.1-mini"

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
"""

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [
    {
        "type": "function",
        "function": price_function
    }
]

ticket_prices = {
    "london":799,
    "paris": 899,
    "tokyo": 1420,
    "sydney": 2999,
    "tel aviv": 1299,
}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)


def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    if tool_call.function.name == "get_ticket_price":
        arguments = json.loads(tool_call.function.arguments)
        city = arguments.get('destination_city')
        price_details = get_ticket_price(city)
        response = {
            "role": "tool",
            "content": price_details,
            "tool_call_id": tool_call.id
        }
    return response

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
            
        
# def get_ticket_price(destination_city):
#     print(f"Tool called for city {destination_city}")
#     price = ticket_prices.get(destination_city.lower(), "Unknown ticket price")
#     return f"The price of a ticket to {destination_city} is {price}"


def chatbot(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools
    )
    
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response = handle_tool_calls(message)
        messages.append(message)
        messages.extend(response)
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools
        )
    
    return response.choices[0].message.content
    


demo = gr.ChatInterface(
    fn=chatbot,
    type="messages"
)

if __name__ == "__main__":
    demo.launch()
    #demo.launch(share=True)