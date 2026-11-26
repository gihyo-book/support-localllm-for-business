import json
import re
def safe_parse_json(text: str) -> dict | None:
    """LLMの出力からJSONを安全に抽出する"""
    # まずそのままパースを試みる
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` のコードブロックが含まれている場合に対応する
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # { } で囲まれた部分を抽出して試みる
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None
