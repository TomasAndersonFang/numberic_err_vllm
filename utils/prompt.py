import re


def format_prompt(task: dict, tokenizer) -> str:
    """Format an MBPP+ task into a prompt string using the tokenizer's chat template."""
    problem = task["prompt"]
    message_content = f"Write a Python function to solve the following problem:\n\n{problem}"

    if getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": message_content}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = message_content

    return prompt


def extract_code(text: str) -> str:
    """Strip markdown code fences from generated text and return raw Python code.

    Defensive: if the input still contains byte-level BPE literal characters
    (Ġ for space, Ċ for newline), the upstream tokenizer decode pipeline did
    not finish properly. We raise loudly instead of returning garbage that
    would silently evaluate to a pass@1 of zero.
    """
    if "\u0120" in text or "\u010a" in text:
        raise RuntimeError(
            "extract_code received text containing byte-level BPE literals "
            "(\u0120 / \u010a). The tokenizer decode step upstream did not "
            "produce a clean string. Check the backend's decode path."
        )

    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text