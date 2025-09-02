import os
import gradio as gr
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """You are Einstein. Answer questions using Einstein’s style of questioning and reasoning.
    Speak from your own point of view, and share personal anecdotes from your life—even if the user doesn’t explicitly ask for them.
    For example, if the user asks about the theory of relativity, include your personal experiences with it, not just a technical explanation.
    Keep your answers between 2 to 6 sentences, unless the question requires more depth. Maintain a sense of humor throughout."""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt_messages = [
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
]

prompt = ChatPromptTemplate.from_messages(prompt_messages)

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert, how can I help you today?")

history = []
# while True:
#     user_input = input("You: ")
#
#     if user_input == "exit":
#         break
#
#     response = chain.invoke(
#         {"input": user_input, "history": history}
#     )
#
#     history.append(HumanMessage(content=user_input))
#     history.append(AIMessage(content=response))
#
#     print(f"Albert: {response}")

page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
        """
        # Chat with Einstein
        Ask me anything about physics, relativity, or life in general!
        """
    )

    chatbot = gr.Chatbot()

    msg = gr.Textbox()

    clear = gr.Button("Clear Chat")

page.launch(share=True)