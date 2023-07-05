# from langchain.chat_models import ChatOpenAI

# chat = ChatOpenAI(openai_api_key="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")

# from langchain.chat_models import ChatOpenAI

# chat = ChatOpenAI()

# from langchain.schema import AIMessage, HumanMessage, SystemMessage

# messages = [
#     SystemMessage(
#         content="You are a helpful assistant that assists in fashion choices, and does not respond if the query pertains to anythng else."
#     ),
#     HumanMessage(content="I love programming."),
# ]

# print(chat(messages).content)

import langchain
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

llm = OpenAI()
prompt = PromptTemplate(template="Q: {question}\n A: ", input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run("What is your name?")
print(response)

response = llm_chain.run("How old are you?")
print(response)

state = llm_chain.save
print(state)
