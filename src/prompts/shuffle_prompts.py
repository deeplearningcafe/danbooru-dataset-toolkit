import random
random.seed(42)
# Define constants for special token IDs for clarity and maintainability.
# These correspond to the CLIP tokenizer used in Stable Diffusion.
START_OF_TEXT_ID = 49406
END_OF_TEXT_ID = 49407
COMMA_ID = 267
# Define the token ID for the combined bracket and comma: '),</w>'
BRACKET_COMMA_ID = 2361
# Use a set for efficient checking of multiple delimiter tokens.
DELIMITER_IDS = {COMMA_ID, BRACKET_COMMA_ID}
MAX_LENGTH = 227

def parse_input_ids(token_ids: list[int], exclude_special_tokens:bool=True):
    # Isolate the core prompt tokens, excluding the start and end tokens.
    # This ensures they are not partt of the shuffling process.
    core_tokens = token_ids
    if exclude_special_tokens:
        core_tokens = core_tokens[1:-1]

    # Parse the tokens into a list of tags. Each tag is a list of tokens.
    tags = []
    current_tag = []
    for token in core_tokens:
        # The comma token acts as a delimiter between tags.
        if token == COMMA_ID:
            # Add the completed tag to our list of tags.
            if current_tag:
                tags.append(current_tag)
            # Reset for the next tag.
            current_tag = []
        elif token == BRACKET_COMMA_ID:
            # Add the completed tag to our list of tags.
            if current_tag:
                current_tag.append(token)
                tags.append(current_tag)
            # Reset for the next tag.
            current_tag = []
        else:
            # Append the token to the tag being built.
            current_tag.append(token)

    # Append the final tag after the loop finishes.
    if current_tag:
        tags.append(current_tag)

    return tags

def reconstruct_input_ids(tags:list[list[int]], include_special_tokens:bool=True, include_comma_last:bool=False):
    shuffled_core_tokens = []
    num_tags = len(tags)
    last_tag = 0 if include_comma_last else 1
    for i, tag in enumerate(tags):
        # Add the tokens of the current tag.
        shuffled_core_tokens.extend(tag)
        # Add a comma after each tag, except for the very last one.
        if i < num_tags - last_tag and tag[-1] != BRACKET_COMMA_ID:
            shuffled_core_tokens.append(COMMA_ID)
    if not include_special_tokens:
        return shuffled_core_tokens
    # Combine the start token, the shuffled core, and the end token.
    return [START_OF_TEXT_ID] + shuffled_core_tokens + [END_OF_TEXT_ID]

def shuffle_prompt_token_ids(
    token_ids: list[int],
    use_dropout: bool = False,
    prompt_len: int = 0,
    include_special_tokens:bool=True,
) -> list[int]:
    """
    Shuffles the tags within a tokenized prompt.

    This function identifies tags as sequences of tokens separated by commas,
    shuffles these tags, and reconstructs the token ID list, keeping the
    start and end tokens in place.

    Args:
        token_ids: A list of integers representing the tokenized prompt.
        use_dropout: If True, enables tag dropout to meet max_length.
        prompt_len: The number of tokens for the prompt.

    Returns:
        A new list of token IDs with the tags shuffled.
    """

    tags = parse_input_ids(token_ids, include_special_tokens)
    # list with the len of each "tag" list of lists
    tags_len = [len(tag) for tag in tags]

    # Apply tag dropout if enabled and necessary.
    if use_dropout:
        removed_tokens = 0
        tokens_2_free = prompt_len - MAX_LENGTH
        while tokens_2_free > 0 and tags:
            # Select a random tag to remove. This ensures that dropout
            # is not biased towards tags at the beginning or end.
            tag_to_remove_index = random.randrange(len(tags))
            tags.pop(tag_to_remove_index)
            removed_tokens += tags_len[tag_to_remove_index]
            tokens_2_free -= removed_tokens

    # Shuffle the list of tags. This is the core randomization step.
    random.shuffle(tags)

    shuffled_tokens = reconstruct_input_ids(tags, include_special_tokens, include_comma_last=not include_special_tokens)
    return shuffled_tokens

def split_upsampled_tags(
        upsampled_tags:str,
        free_tokens:int,
        tokenizer,
        return_last_token:bool=False,
    ) -> str:
    if free_tokens < 1:
        if return_last_token:
            return "", None
        return ""

    upsampled_tokens= tokenizer(
        upsampled_tags,
        padding=False, # Don't pad here, just count tokens
        truncation=False, # Don't truncate as we want length
        return_length=True,
    )
    # if more than enough space then don't do anything
    if (upsampled_tokens["length"]-2) < free_tokens:
        if return_last_token:
            # The last content token is at index -2, before the EOS token.
            last_token = upsampled_tokens["input_ids"][-2]
            return upsampled_tags, last_token
        else:
            return upsampled_tags

    tags = parse_input_ids(upsampled_tokens["input_ids"])
    included_tags = []
    remainig_tokens = free_tokens
    for tag in tags:
        tag_len = len(tag)
        if tag_len < remainig_tokens:
            included_tags.append(tag)
            remainig_tokens -= tag_len
        else:
            break

    # now convert the tags to str
    sliced_input_ids = reconstruct_input_ids(included_tags)
    # we need to decode and then split for removing the blanks after each tag
    # and remove the comma of the last element if "),"
    # there is no need to remove the blanks as the prompts are the same when tokenizing
    decoded_sliced_input_ids = tokenizer.decode(
        sliced_input_ids, skip_special_tokens=True
    )
    if return_last_token:
        # Ensure we don't access an empty list if no tags were included.
        last_token = sliced_input_ids[-2] if sliced_input_ids else None
        return decoded_sliced_input_ids, last_token
    return decoded_sliced_input_ids

