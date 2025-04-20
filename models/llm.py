import openai
from config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, PROMPT_SAFETY_DIRECTIVE

openai.api_key = OPENAI_API_KEY

def call_llm(prompt: str) -> str:
    """
    Calls GPT-4 (or the configured model) using the OpenAI ChatCompletion API.
    PROMPT_SAFETY_DIRECTIVE is added to the prompt to ensure safety.
    """
    safe_prompt = f"{PROMPT_SAFETY_DIRECTIVE}\n{prompt}"
    response = openai.ChatCompletion.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a production-ready advanced code assistant. Respond with safe, clear and thorough answers."},
            {"role": "user", "content": safe_prompt}
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return response["choices"][0]["message"]["content"] 