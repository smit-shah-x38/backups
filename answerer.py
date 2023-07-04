from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(openai_api_key="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")

from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()

from langchain.schema import AIMessage, HumanMessage, SystemMessage

batch_messages = [
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ],
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love artificial intelligence."),
    ],
]
result = chat.generate(batch_messages)
print(result)
