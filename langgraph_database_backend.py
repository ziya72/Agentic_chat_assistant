from langgraph.graph import StateGraph, START, END, add_messages
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

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

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# ---- 4. Graph ----
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)