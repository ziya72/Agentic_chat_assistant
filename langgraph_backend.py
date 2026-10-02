from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Load GROQ_API_KEY (and any other secrets) from .env into the environment.
# Must happen before ChatGroq() is constructed, since it reads the key at init time.
load_dotenv()

# ---- 1. State schema ----
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---- 2. LLM ----
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)

# ---- 3. Node(s) ----
def chat_node(state: ChatState) -> ChatState:
    # take user query from state
    messages = state['messages']

    # send to llm
    response = llm.invoke(messages)

    # response store state
    return {'messages': [response]}

# ---- 4. Graph ----
checkpointer = MemorySaver()

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {'configurable': {'thread_id': 'thread-1 '}}

response = chatbot.invoke(
    {'messages': [HumanMessage(content='Hi my name is nitish')]},
    config = CONFIG,
)

print(chatbot.get_state(config=CONFIG).values['messages'])