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
from flask import Flask, request, jsonify
import langchain
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

llm = OpenAI()
conversation_history = [
    "You are a helpful assistant that specializes in fashion shopping. Only answer my questions if they are related to fashion, otherwise answer with Please ask a relevant question"
]


@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():
    global conversation_history
    question = request.json["question"]
    prompt = PromptTemplate(
        template="\n".join(conversation_history) + "\nQ: {question}\n A: ",
        input_variables=["question"],
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.run(question)
    conversation_history.append(f"Q: {question}\nA: {response}")
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
