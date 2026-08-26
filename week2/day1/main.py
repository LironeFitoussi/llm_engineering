# LangChain Testing
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pathlib import Path

# Get the directory of the current script
load_dotenv()

llm = ChatOpenAI(model="gpt-5-mini")


# ? Joke Test
tell_a_joke = [
    {"role": "user", "content": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering"},
]

# response = llm.invoke(tell_a_joke)

# print(response.content)

#? Extrnal file Test
def get_text():
    BASE_DIR = Path(__file__).resolve().parents[1]
    # print(BASE_DIR)
    with open(f"{BASE_DIR}/hamlet.txt", "r", encoding="utf-8") as f:
        text = f.read()
        return text


question = [{"role": "user", "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"}]

question[0]["content"] += "\n\nFor context, here is the entire text of Hamlet:\n\n"+get_text()

print("==================")
print(question)
print("==================")
response = llm.invoke(question)
print(response.content)