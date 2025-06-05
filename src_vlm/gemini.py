import concurrent.futures
from typing import Any, List, Optional, Tuple

from google import genai
from google.genai import types
from loguru import logger
from pydantic_ai.usage import Usage


class GeminiClient:
    def __init__(self, model_name: str, api_key: str, **kwargs: Any):
        self._client = genai.Client(
            api_key=api_key,
        )
        self._model_name = model_name
        self._get_model_info()

    def _get_model_info(self):
        """Get model info to check if the model is supported."""
        try:
            self._client.models.get(model=self._model_name)
        except Exception as e:
            raise ValueError(f"Model {self._model_name} is not supported.") from e

    def request(
        self,
        messages: List[Any],
        **kwargs: Any,
    ) -> Optional[Tuple[str, Usage]]:
        """Send a synchronous request to the Gemini API."""
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=messages,
                config=types.GenerateContentConfig(
                    **kwargs,
                ),
            )

            # NOTE: Response tokens include both output tokens and thinking tokens
            usage = Usage(
                requests=1,
                request_tokens=response.usage_metadata.prompt_token_count,
                response_tokens=response.usage_metadata.total_token_count
                - response.usage_metadata.prompt_token_count,
                total_tokens=response.usage_metadata.total_token_count,
                details={
                    "thinking_tokens": response.usage_metadata.thoughts_token_count
                    if response.usage_metadata.thoughts_token_count
                    else 0,
                },
            )
            return response.text, usage
        except Exception as e:
            logger.error(f"Unexpected error during GEMINI API request: {e}")
            return None

    def request_batch(
        self,
        batch_messages: List[List[Any]],
        max_concurrency: int = 8,
        **kwargs: Any,
    ) -> list[Optional[Tuple[str, Usage]]]:
        """Send a batch (synchronous) request to the Gemini API."""

        results = [None] * len(batch_messages)

        def process_request(idx, messages):
            logger.debug(f"Sending request {idx}")
            result = self.request(
                messages=messages,
                **kwargs,
            )
            return idx, result

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrency
        ) as executor:
            # Submit all tasks and create a future-to-index mapping
            futures = [
                executor.submit(process_request, i, messages)
                for i, messages in enumerate(batch_messages)
            ]

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception as e:
                    logger.error(f"Error in thread processing request: {e}")

        return results
