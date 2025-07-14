import re


def extract_block(text: str, tag: str | None = None):
    """从文本中提取被 ``` 包裹的代码块，如果是多个代码块，返回最后一个"""
    pattern = r"```(\w+)\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)

    filtered_matches = [code for lang, code in matches if tag is None or lang == tag]
    return filtered_matches[-1] if filtered_matches else text


if __name__ == "__main__":
    with open("tmp.txt", "r") as f:
        text = f.read()
    code_blocks = extract_block(text)
    print(code_blocks)
