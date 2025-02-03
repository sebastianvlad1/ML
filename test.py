# Simplified working version:
from llama_cpp import Llama

CUSTOM_DATA = """Sebi loves Delia the most. Sebi is Delia's lover. They have a dog called Bixbi."""

model_path = "Llama-3.2-3B-Instruct-Q5_K_M.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_gpu_layers=-1,
    n_threads=8,
    verbose=True
)

prompt = f"""### Instruction:
You are a fact-bot. Answer ONLY using this data: {CUSTOM_DATA}
If unsure, say "I don't know."

### User:
Who is Sebi?

### Assistant:
"""


response = llm(
    prompt,
    max_tokens=200,
    temperature=0.1,
    stop=["<|eot_id|>"]
)

print(response["choices"][0]["text"])