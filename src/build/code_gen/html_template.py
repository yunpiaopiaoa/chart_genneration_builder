from pathlib import Path
import re

template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ECharts</title>
    <script src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
</head>
<body style="display: flex;justify-content: center;align-items: center;height: 100vh;">
    <div id="chart" style="width: 1500px; height: 1050px;"></div>
    <script>
{script}
    </script>
</body>
</html>
"""


class HtmlTemplate:
    def __init__(self):
        self.html_template = template
        self.pattern = re.compile(r'getElementById\((["\'])(.*?)\1\)')

    def instance(self, script:str):
        match = re.search(self.pattern, script)
        if match is None:#没有调用getElementById函数，可能只有js配置，或者误传入pycharts代码
            print(script)
        id = match.group(2)
        html_template = self.html_template.replace('<div id="chart"', f'<div id="{id}"')
        script="\n".join(" "*8 + line.strip() for line in script.splitlines())
        html = html_template.format(script=script)
        return html
