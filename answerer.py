# # Import the necessary modules
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # Load the tokenizer and the model
# tokenizer = AutoTokenizer.from_pretrained("AlekseyKorshuk/vicuna-7b", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("AlekseyKorshuk/vicuna-7b")

# # Define the instruction and the user message
# instruction = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
# user_message = "What is your favorite book?"

# # Encode the instruction and the user message as input ids
# input_ids = tokenizer.encode(instruction.format(user_message), return_tensors="pt")

# # Generate a response from the model
# output_ids = model.generate(input_ids, max_length=100, do_sample=True)

# # Decode the output ids as text
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # Print the output text
# print(output_text)

import openai

openai.api_key = (
    "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"  # Not support yet
)
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.3"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
    model=model, messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
