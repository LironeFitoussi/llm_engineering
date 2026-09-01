
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

image_function = {
    "name": "generate_destination_image",
    "description": (
        "Generate and show the customer a vibrant pop-art image of a destination city. "
        "Call this when the customer asks to see a city, asks what somewhere looks like, "
        "or is weighing up destinations and a picture would help them decide. "
        "Do not call it for a plain price question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city to depict in the image",
            },
        },
        "required": ["city"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": image_function},
]
