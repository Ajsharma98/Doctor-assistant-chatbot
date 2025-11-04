from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# 1. Load your Hugging Face API token
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 2. Initialize the LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=200,
    huggingfacehub_api_token=HF_TOKEN,
    stop_sequences=["User query:"]
)

# 3. Stronger, clearer prompt
prompt_text = """You are a helpful assistant.
Your task is to generate 4 distinct paraphrases of the user's query.
Each paraphrase must keep the same meaning but use different phrasing.

Write exactly four paraphrased versions, each on a new line, numbered 1 to 4.

Example:
User query: "what is AI"
Output:
1. What does AI mean?
2. What is the definition of AI?
3. How is AI defined?
4. What does the term AI refer to?

Now do the same for the following:
User query: "{user_query}"
Output:
"""

prompt = PromptTemplate(
    input_variables=["user_query"],
    template=prompt_text
)

# 4. Chain the prompt to the model
chain = prompt | llm

# 5. Run the chain
result = chain.invoke({"user_query": "what is Hb in medical term"})

# 6. Extract generated text
if isinstance(result, dict) and "generated_text" in result:
    print(result["generated_text"].strip())
else:
    print(result)
