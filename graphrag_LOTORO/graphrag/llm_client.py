"""
llm_client.py
=============
Thin wrapper around LLM APIs providing chat completions and text embeddings
for the GraphRAG pipeline. Supports multiple providers via a unified interface.

Supported providers:
  - openai   : OpenAI API (gpt-4o, gpt-4o-mini, text-embedding-3-small, ...)
  - google   : Google Gemini via OpenAI-compatible endpoint
               (gemini-2.5-flash, gemini-2.0-flash, ...)
  - anthropic: Anthropic Claude via OpenAI-compatible endpoint
               (claude-sonnet-4-5, claude-haiku-4-5, ...)

Provider is set per model role in config.yaml:
    llm:
      answer_model:    "gpt-4o"
      answer_provider: "openai"          # openai | google | anthropic
      answer_api_key:  "sk-..."
      report_model:    "gemini-2.5-flash"
      report_provider: "google"
      report_api_key:  "AIzaSy..."
      embedding_model: "text-embedding-3-small"
      embedding_provider: "openai"
      embedding_api_key:  "sk-..."

If a role-specific key is absent, falls back to llm.api_key.

All calls are retried with exponential backoff on API errors and JSON
parse failures. Temperature 0.0 for deterministic, reproducible outputs.

"""

"""
llm_client.py
=============
Thin wrapper around LLM APIs providing chat completions and text embeddings
for the GraphRAG pipeline. Extends the original single-provider design to
support multiple providers via a unified OpenAI-compatible interface.

Supported providers: openai | google | anthropic
Provider is set per model role in config.yaml:
    llm:
      answer_model:    "gpt-4o"
      answer_provider: "openai"
      answer_api_key:  "sk-..."
      report_model:    "gemini-2.0-flash"
      report_provider: "google"
      report_api_key:  "AIzaSy..."
      embedding_model:    "text-embedding-3-small"
      embedding_provider: "openai"
      embedding_api_key:  "sk-..."

If role-specific keys are absent, falls back to llm.api_key.
Environment variables take precedence: OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY.
All other behaviour (retry logic, backoff, JSON mode) is unchanged.

Dependencies
------------
- openai >= 1.30.0  (used for all providers via compatibility layer)
"""

import json
import logging
import os
import time

from openai import OpenAI

logger = logging.getLogger(__name__)

# Provider base URLs — all accessed via OpenAI-compatible REST interface
_PROVIDER_BASE_URLS = {
    "openai":    None,
    "google":    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "anthropic": "https://api.anthropic.com/v1/",
}

# Environment variable names per provider
_PROVIDER_ENV_VARS = {
    "openai":    "OPENAI_API_KEY",
    "google":    "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}


def _make_client(provider: str, api_key: str) -> OpenAI:
    """Creates an OpenAI-compatible client for the given provider."""
    base_url = _PROVIDER_BASE_URLS.get(provider)
    env_key  = os.getenv(_PROVIDER_ENV_VARS.get(provider, "OPENAI_API_KEY"))
    key      = env_key or api_key
    if base_url:
        return OpenAI(api_key=key, base_url=base_url)
    return OpenAI(api_key=key)


class LLMClient:

    def __init__(self, config: dict):
        cfg            = config["llm"]
        self.cfg       = cfg
        fallback_key   = cfg.get("api_key", "")

        # Build one client per slot — same three slots as original
        def _slot(model_key, provider_key, api_key_key):
            provider = cfg.get(provider_key, "openai").lower()
            api_key  = cfg.get(api_key_key, fallback_key)
            return {
                "model":    cfg[model_key],
                "provider": provider,
                "client":   _make_client(provider, api_key),
            }

        self._slots = {
            "answer": _slot("answer_model",    "answer_provider",    "answer_api_key"),
            "report": _slot("report_model",    "report_provider",    "report_api_key"),
            "embed":  _slot("embedding_model", "embedding_provider", "embedding_api_key"),
        }

        for role, slot in self._slots.items():
            logger.info(
                f"LLMClient [{role}]: model={slot['model']} "
                f"provider={slot['provider']} "
                f"key={'SET' if slot['client'].api_key else 'MISSING'}"
            )

    def chat(
        self,
        system_prompt : str,
        user_prompt   : str,
        model_role    : str        = "answer",
        expect_json   : bool       = True,
        max_tokens    : int | None = None,
    ) -> dict | str:
        """
        Args:
            system_prompt : System instruction string.
            user_prompt   : User message / context.
            model_role    : One of 'report', 'answer'.
            expect_json   : If True, enforces json_object mode and parses output.
            max_tokens    : Override default max_tokens from config.

        Returns:
            Parsed dict if expect_json=True, raw string otherwise.

        Raises:
            RuntimeError if all retries are exhausted.
        """
        slot     = self._slots[model_role]
        model    = slot["model"]
        provider = slot["provider"]
        client   = slot["client"]
        max_tok  = max_tokens or self.cfg["max_tokens"]
        last_err = None

        for attempt in range(self.cfg["max_retries"]):
            try:
                kwargs = dict(
                    model       = model,
                    messages    = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature = self.cfg["temperature"],
                    max_tokens  = max_tok,
                    timeout     = self.cfg["timeout"],
                )
                # JSON mode supported by openai and google, not anthropic
                if expect_json and provider != "anthropic":
                    if self.cfg.get("model_supports_json", True):
                        kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if expect_json:
                    return json.loads(content)
                return content

            except json.JSONDecodeError as exc:
                last_err = exc
                logger.warning(
                    f"[LLMClient:{model_role}] JSON parse error "
                    f"(attempt {attempt + 1}/{self.cfg['max_retries']}): {exc}"
                )
            except Exception as exc:
                last_err = exc
                logger.warning(
                    f"[LLMClient:{model_role}] API error "
                    f"(attempt {attempt + 1}/{self.cfg['max_retries']}): {exc}"
                )

            if attempt < self.cfg["max_retries"] - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"[LLMClient:{model_role}] All {self.cfg['max_retries']} retries exhausted. "
            f"Last error: {last_err}"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Returns embeddings for a list of texts.
        Called only by the vector fallback branch in search_engine.

        Returns:
            List of float vectors, one per input string.

        Raises:
            RuntimeError if all retries are exhausted.
        """
        slot     = self._slots["embed"]
        model    = slot["model"]
        client   = slot["client"]
        last_err = None

        for attempt in range(self.cfg["max_retries"]):
            try:
                response = client.embeddings.create(model=model, input=texts)
                return [item.embedding for item in response.data]

            except Exception as exc:
                last_err = exc
                logger.warning(
                    f"[LLMClient:embed] Embedding error "
                    f"(attempt {attempt + 1}/{self.cfg['max_retries']}): {exc}"
                )

            if attempt < self.cfg["max_retries"] - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(
            f"[LLMClient:embed] All {self.cfg['max_retries']} retries exhausted. "
            f"Last error: {last_err}"
        )