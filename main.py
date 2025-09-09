import os
import gradio as gr
from dotenv import load_dotenv
from typing import List, Dict, Tuple

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

def chat(user_input: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """
        Handles one round of user input and model response.
        Converts Gradio chat_history to LangChain format,
        queries the model, and appends the new messages.
    """

    langchain_history = []

    for item in chat_history:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))

    response = chain.invoke(
        {"input": user_input, "history": langchain_history}
    )

    return "", chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": response}
    ]

def clear_chat() -> Tuple[str, List[Dict[str, str]]]:
    return "", []

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

    chatbot = gr.Chatbot(
        type='messages',
        avatar_images=(None, 'einstein.png'),
        show_label=False
    )

    msg = gr.Textbox(
        show_label=False,
        placeholder="Enter your message here and press Enter",
        container=False
    )

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)