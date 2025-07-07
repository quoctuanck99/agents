from __future__ import annotations
import regex as re
from collections.abc import Awaitable
from typing import Callable, Union

from livekit.agents import llm
from livekit.agents.llm.tool_context import (
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from openai.types.chat import ChatCompletionToolParam

AsyncAzureADTokenProvider = Callable[[], Union[str, Awaitable[str]]]


def get_base_url(base_url: str | None) -> str:
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return base_url


def to_fnc_ctx(
    fnc_ctx: list[llm.FunctionTool | llm.RawFunctionTool],
) -> list[ChatCompletionToolParam]:
    tools: list[ChatCompletionToolParam] = []
    for fnc in fnc_ctx:
        if is_raw_function_tool(fnc):
            info = get_raw_function_info(fnc)
            tools.append(
                {
                    "type": "function",
                    "function": info.raw_schema,  # type: ignore
                }
            )
        elif is_function_tool(fnc):
            tools.append(llm.utils.build_strict_openai_schema(fnc))  # type: ignore

    return tools

def filter_transcripts_based_on_log_probs(
        log_probs: list,
        single_word_threshold: float = -1.5,  # Threshold for single-token sentences
        token_threshold: float = -0.15,  # Threshold for individual token logprob (except first token)
        avg_logprob_threshold: float = -0.5,  # Threshold for average logprob (excluding first token)
        min_tokens_for_avg: int = 4,  # Minimum tokens to apply average logprob threshold
        short_sentence_token_ratio: float = 0.8,  # Token ratio threshold for short sentences (2-3 tokens)
        long_sentence_token_ratio: float = 0.6,  # Token ratio threshold for long sentences (>=4 tokens)
        first_token_threshold: float = -2.0  # Separate threshold for the first token
) -> bool:
    """
    Filter speech-to-text results based on logprobs, handling low logprob of the first token.

    Args:
        log_probs: List of tokens and their logprobs.
        single_word_threshold: Logprob threshold for single-token sentences.
        token_threshold: Logprob threshold for individual tokens (except first token).
        avg_logprob_threshold: Threshold for average logprob (excluding first token).
        min_tokens_for_avg: Minimum number of tokens to apply average logprob threshold.
        short_sentence_token_ratio: Token ratio threshold for short sentences (2-3 tokens).
        long_sentence_token_ratio: Token ratio threshold for long sentences (>=4 tokens).
        first_token_threshold: Logprob threshold for the first token.

    Returns:
        bool: True if transcript is valid, False otherwise.
    """
    if not log_probs:
        logger.info("Empty log_probs, rejecting transcript.")
        return False

    # Initialize variables
    accepted_prob_cnt = 0
    total_valid = 0
    logprob_sum = 0.0
    valid_tokens = []

    # Process each token
    for i, prob in enumerate(log_probs):
        token = prob["token"]
        logprob = prob["logprob"]

        # Skip special characters (punctuation)
        if not re.sub(r"\p{P}+", "", token):
            logger.info(f"Ignore character '{token}' | {logprob}")
            continue

        total_valid += 1
        valid_tokens.append((token, logprob))

        # Check token logprob
        if i == 0 and total_valid == 1:
            # First token
            threshold = first_token_threshold
        else:
            threshold = token_threshold

        if logprob > threshold:
            accepted_prob_cnt += 1
            logger.info(f"{token} | {logprob}: ok (threshold: {threshold})")
        else:
            logger.info(f"{token} | {logprob}: notok (threshold: {threshold})")

    # Calculate token ratio and average logprob (excluding first token if >= 2 tokens)
    if total_valid == 0:
        logger.info("No valid tokens, rejecting transcript.")
        return False

    if total_valid == 1:
        # Single-token case: require logprob > single_word_threshold
        logprob = valid_tokens[0][1]
        logger.info(f"Single token case, logprob: {logprob}, threshold: {single_word_threshold}")
        return logprob > single_word_threshold

    # Calculate average logprob and ratio (excluding first token)
    tokens_to_consider = valid_tokens[1:] if len(valid_tokens) > 1 else valid_tokens
    accepted_prob_cnt_without_first = sum(1 for _, logprob in tokens_to_consider if logprob > token_threshold)
    total_valid_without_first = len(tokens_to_consider)
    logprob_sum_without_first = sum(logprob for _, logprob in tokens_to_consider)

    accepted_prob_percent = (
        accepted_prob_cnt_without_first / total_valid_without_first
        if total_valid_without_first > 0 else 0
    )
    avg_logprob = (
        logprob_sum_without_first / total_valid_without_first
        if total_valid_without_first > 0 else float('-inf')
    )

    # Check first token separately
    first_token_ok = valid_tokens[0][1] > first_token_threshold

    logger.info(f"Accepted prob percent (without first token): {accepted_prob_percent:.2f}")
    logger.info(f"Average logprob (without first token): {avg_logprob:.2f}")
    logger.info(f"Total valid tokens: {total_valid}")
    logger.info(f"First token logprob: {valid_tokens[0][1]}, ok: {first_token_ok}")

    # Decision rules
    if total_valid <= 3:
        # Short sentence (2-3 tokens): check first token and ratio of remaining tokens
        token_ratio_ok = accepted_prob_percent >= short_sentence_token_ratio
        return first_token_ok and token_ratio_ok

    # Long sentence (>=4 tokens): check first token, token ratio, and average logprob
    token_ratio_ok = accepted_prob_percent >= long_sentence_token_ratio
    avg_logprob_ok = avg_logprob > avg_logprob_threshold if total_valid >= min_tokens_for_avg else True

    logger.info(f"Token ratio ok: {token_ratio_ok}, Avg logprob ok: {avg_logprob_ok}, First token ok: {first_token_ok}")
    return first_token_ok and token_ratio_ok and avg_logprob_ok
