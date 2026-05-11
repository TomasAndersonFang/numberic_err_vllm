import json
import tempfile
import os

from utils.prompt import extract_code, format_prompt
from utils.eval_utils import parse_evalplus_output


class MockTokenizerNoTemplate:
    chat_template = None


class MockTokenizerWithTemplate:
    chat_template = "some_template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"


def test_extract_code_with_python_fence():
    text = '```python\ndef foo():\n    return 42\n```'
    result = extract_code(text)
    assert "```" not in result
    assert "def foo():" in result


def test_extract_code_with_plain_fence():
    text = '```\ndef bar():\n    pass\n```'
    result = extract_code(text)
    assert "```" not in result
    assert "def bar():" in result


def test_extract_code_no_fence():
    text = "def baz():\n    return 1"
    result = extract_code(text)
    assert result == text


def test_format_prompt_no_chat_template():
    task = {"prompt": "Write a function that adds two numbers."}
    tokenizer = MockTokenizerNoTemplate()
    result = format_prompt(task, tokenizer)
    assert len(result) > 0
    assert "Write a function that adds two numbers." in result


def test_format_prompt_with_chat_template():
    task = {"prompt": "Write a function that adds two numbers."}
    tokenizer = MockTokenizerWithTemplate()
    result = format_prompt(task, tokenizer)
    assert len(result) > 0
    assert "Write a function that adds two numbers." in result
    assert "<|user|>" in result


def test_parse_evalplus_output_pass():
    data = {
        "eval": {
            "Mbpp/2": [{"base_status": "pass", "plus_status": "pass"}],
            "Mbpp/3": [{"base_status": "pass", "plus_status": "failed"}],
            "Mbpp/4": [{"base_status": "failed", "plus_status": "failed"}],
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    try:
        result = parse_evalplus_output(path)
        assert result["Mbpp/2"] == "pass"
        assert result["Mbpp/3"] == "fail"
        assert result["Mbpp/4"] == "fail"
    finally:
        os.unlink(path)
