import openai
import ollama
import logging
import time
from config import settings # Import the settings object

logger = logging.getLogger(__name__)

# Initialize LLM client based on provider
llm_client = None

if settings.LLM_PROVIDER == "openai":
    if not settings.OPENAI_API_KEY:
        logger.error("OpenAI LLM provider selected, but OPENAI_API_KEY is not set.")
        # raise ValueError("OPENAI_API_KEY is required for the OpenAI provider.") # Option: Fail fast
    else:
        try:
            llm_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            # Health Check: Try listing models (lightweight check)
            llm_client.models.list()
            logger.info("Successfully connected to OpenAI LLM provider.")
        except openai.AuthenticationError:
            logger.error("OpenAI API key is invalid. Please check your OPENAI_API_KEY.")
            llm_client = None
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI LLM provider: {e}")
            llm_client = None
elif settings.LLM_PROVIDER == "ollama":
    try:
        llm_client = ollama.Client(host=settings.LLM_BASE_URL)
        # Health Check: List local models
        llm_client.list()
        logger.info(f"Successfully connected to Ollama LLM provider at {settings.LLM_BASE_URL}.")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama LLM provider at {settings.LLM_BASE_URL}: {e}")
        llm_client = None
else:
    logger.error(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}. Only 'openai' and 'ollama' are currently supported.")
    # raise ValueError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}") # Option: Fail fast

def call_llm(prompt: str, max_retries=2, retry_delay=3) -> str:
    """
    Calls the configured LLM provider (OpenAI or Ollama) with the given prompt.
    Includes safety directives and basic retry logic for transient errors.
    Raises RuntimeError if the call fails after retries or provider is unavailable.
    """
    if not llm_client:
        raise RuntimeError(f"LLM provider '{settings.LLM_PROVIDER}' is not available or failed to initialize.")

    safe_prompt = f"{settings.PROMPT_SAFETY_DIRECTIVE}\n{prompt}"
    messages = [
        {"role": "system", "content": "You are a production-ready advanced code assistant. Respond with safe, clear and thorough answers."},
        {"role": "user", "content": safe_prompt}
    ]

    for attempt in range(max_retries + 1):
        try:
            if settings.LLM_PROVIDER == "openai":
                response = llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                return response.choices[0].message.content
            elif settings.LLM_PROVIDER == "ollama":
                response = llm_client.chat(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    options={
                        'temperature': settings.LLM_TEMPERATURE,
                        # Ollama doesn't have a direct max_tokens equivalent in the same way
                        # 'num_predict': settings.LLM_MAX_TOKENS # Use this if needed, but be aware of model limits
                    }
                )
                return response['message']['content']
            else:
                # This case should ideally be prevented by the initialization check
                raise RuntimeError(f"LLM provider '{settings.LLM_PROVIDER}' is not supported.")

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}. Check API key.")
            raise RuntimeError(f"OpenAI Authentication Failed: {e}") from e
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit Error: {e}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise RuntimeError(f"OpenAI Rate Limit Exceeded after retries: {e}") from e
            time.sleep(retry_delay)
        except (openai.APIConnectionError, openai.APITimeoutError, TimeoutError) as e:
            logger.warning(f"LLM Connection/Timeout Error: {e}. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
            if attempt == max_retries:
                raise RuntimeError(f"LLM Connection Failed after retries: {e}") from e
            time.sleep(retry_delay)
        except ollama.ResponseError as e:
             # Ollama errors might include model not found, connection issues, etc.
            logger.error(f"Ollama Error: {e.error} (Status code: {e.status_code})")
            # Decide if retry is appropriate based on error type, e.g., not for model not found
            if "connection refused" in e.error.lower() and attempt < max_retries:
                 logger.warning(f"Ollama connection refused. Retrying in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
                 time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Ollama API call failed: {e.error}") from e
        except Exception as e:
            logger.error(f"Unexpected LLM call error ({type(e).__name__}): {e}")
            # Generic retry for unexpected errors
            if attempt < max_retries:
                logger.warning(f"Retrying unexpected error in {retry_delay}s... (Attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"LLM call failed due to unexpected error after retries: {e}") from e
                
    # Should not be reachable if logic is correct, but acts as a safeguard
    raise RuntimeError("LLM call failed after all retries.") 