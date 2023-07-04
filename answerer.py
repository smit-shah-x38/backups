from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(openai_api_key="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")

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()

from langchain.schema import AIMessage, HumanMessage, SystemMessage

messages = [
    SystemMessage(
        content="You are a helpful assistant that assists in fashion choices, and does not respond if the query pertains to anythng else."
    ),
    HumanMessage(content="I love programming."),
]

print(chat(messages).content)
