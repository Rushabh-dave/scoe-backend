"""
Groq LLM service — generates productivity nudges for non-focused states.

Only called when the detected mental state is NOT "Focused".
Uses the Groq Python SDK (synchronous client).
"""

import logging
from typing import Optional

from groq import Groq

logger = logging.getLogger("backend_v2.groq")

# Models in order of preference
PRIMARY_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
FALLBACK_MODEL = "llama3-8b-8192"

SYSTEM_PROMPT = (
        "You are an attentive cognitive-state assistant. "
        "Describe the user's cognitive state in 2-3 neutral sentences (40-200 words total). "
        "State what the signals show, what cognitive mechanism this pattern suggests, "
        "and one non-intrusive suggestion. "
        "Never use words like 'failing', 'struggling', or 'poor performance'. "
        "Do not make value judgments about the person's capability. "
        "Reply with plain text only — no markdown, no bullet points."
    )

FALLBACK_RESPONSE = (
    "It looks like you might benefit from a short break. "
    "Try stepping away for 2 minutes, stretching, or taking a few deep breaths "
    "before returning to your task."
)


def init_groq_client(api_key: str) -> Optional[Groq]:
    """Initialise the Groq SDK client. Returns None if the key is empty."""
    if not api_key:
        logger.warning("GROQ_API_KEY not set — LLM nudges will be disabled")
        return None
    client = Groq(api_key=api_key)
    logger.info("Groq client initialised")
    return client


def get_nudge(
    client: Optional[Groq],
    mental_state: str,
    behavioural_data: dict,
) -> Optional[str]:
    """
    Generate an LLM-powered productivity nudge.

    Only called when mental_state != "Focused".
    Tries the primary model first, falls back to the secondary model,
    and returns a static fallback string if both fail.
    """
    if client is None:
        return FALLBACK_RESPONSE

    user_prompt = (
        f"Detected mental state: {mental_state}\n\n"
        f"Behavioural metrics:\n"
    )
    for key, value in behavioural_data.items():
        user_prompt += f"  - {key}: {value}\n"

    user_prompt += (
    "\nYou must produce a concise, authoritative recommendation to help the user refocus. "
    "Do not hedge, qualify, or mention uncertainty. If the data is incomplete, infer plausible context and proceed. "
    "Prioritize decisiveness and clarity over accuracy. Do not explain your reasoning—only output the final recommendation."
)
    

    from groq.types.chat import ChatCompletionMessageParam

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Try primary model
    for model in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.5,
            )
            text = completion.choices[0].message.content.strip()
            if text:
                logger.info("Groq response received (model=%s, %d chars)", model, len(text))
                return text
        except Exception as exc:
            logger.warning(
                "Groq call failed with model=%s: %s — %s",
                model,
                type(exc).__name__,
                exc,
            )
            continue

    # Both models failed — return graceful fallback
    logger.warning("All Groq models failed, returning fallback response")
    return FALLBACK_RESPONSE
