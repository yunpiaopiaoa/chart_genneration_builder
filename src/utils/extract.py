import re

def extract_block(text:str):
    """从文本中提取被 ``` 包裹的代码块"""
    pattern = r'```(?:[a-zA-Z0-9_+-]*\n)?(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text

if __name__ == '__main__':
    with open('tmp.txt', 'r') as f:
        text = f.read()
    code_blocks = extract_block(text)
    print(code_blocks)