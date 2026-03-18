import os
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("NO TOKEN")
    exit(1)

client = InferenceClient("meta-llama/Meta-Llama-3-70B-Instruct", token=hf_token)
try:
    response = client.chat_completion(
        messages=[{"role": "user", "content": "Say 'hello'"}],
        max_tokens=5,
        temperature=0.0
    )
    print("SUCCESS:", response.choices[0].message.content)
except Exception as e:
    print("ERROR:", e)
