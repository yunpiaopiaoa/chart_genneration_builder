# from playwright.async_api import async_playwright
from http.server import SimpleHTTPRequestHandler
import os
from pathlib import Path
import re
import socketserver
import threading
from playwright.sync_api import sync_playwright

from .base_img_generator import BaseImgGenerator


class QuietServer(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass
class MyServer(socketserver.TCPServer):
    allow_reuse_address = True  # 允许程序重新运行后重新绑定端口

class EchartsImgGenerator(BaseImgGenerator):
    """只使用同步"""

    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.port = 8000
        self.httpd = MyServer(("", self.port), QuietServer)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.start()
        print(f"EchartsImgGenerator:服务器已启动，监听端口 {self.port}")
        curdir = Path(__file__).parent
        with (curdir.parent  / "code_gen" / "template.html").open(
            "r", encoding="utf-8"
        ) as f:
            self.html_template = f.read()
    def cleanup(self):
        if self.browser is not None:
            self.browser.close()
            self.browser = None
        if self.playwright is not None:
            self.playwright.stop()
            self.playwright = None
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        print(f"EchartsImgGenerator:服务器已关闭")
        if self.server_thread:
            self.server_thread.join()
        print(f"EchartsImgGenerator:服务器线程已结束")

    def generate_img(self, code: str, save_path: str,width=1000,height=800):
        """借助无头浏览器渲染 ECharts 图表并保存为图片"""

        if not re.search(r"<html>", code):#如果不存在html标签，说明是js代码，需要使用模板包裹起来
            code=self.html_template.format(script=code)
            match = re.search(r'getElementById\((["\'])(.*?)\1\)', code)
            id = match.group(2)#找到图表挂载dom元素id
            code = code.replace('<div id="chart"', f'<div id="{id}"')
        page = self.browser.new_page(viewport={"width": width, "height": height})
        content = re.sub(
            r'src="[^"]*"',
            f'src="http://localhost:{self.port}/lib/echarts.min.js"',
            code,
        )
        page.set_content(content, wait_until="load")
        pattern = r'getElementById\((["\'])(.*?)\1\)'
        match = re.search(pattern, code)
        id = match.group(2)
        # WARNING：找不到一个通用的方法确定页面完全加载,暂定3000ms
        page.wait_for_selector(f"#{id}", state="visible")
        page.wait_for_timeout(3000)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        element=page.query_selector(f"id={id}")
        element.screenshot(path=save_path)
        page.close()



# class EchartsImgGenerator(BaseImgGenerator):
#     """使用异步"""
#     def __init__(self):
#         self.playwright = None
#         self.browser = None
#         self.page = None

#     async def create_source(self):
#         self.playwright = await async_playwright().start()
#         self.browser = await self.playwright.chromium.launch(headless=True)

#     async def cleanup(self):
#         """对象析构时释放浏览器资源"""
#         if self.browser is not None:
#             await self.browser.close()
#             self.browser = None
#         if self.playwright is not None:
#             await self.playwright.stop()
#             self.playwright = None
#     def __del__(self):
#         """对象析构时记录警告，提示需要显式调用 cleanup"""
#         if any([self.browser, self.playwright]):
#             import warnings
#             warnings.warn("Resources were not cleaned up properly. Call cleanup() before deletion.", RuntimeWarning)
#     async def generate_img(self, code: str, save_path: str):
#         if self.browser is None:
#             await self.create_source()
#         page = await self.browser.new_page(viewport={"width": 1000, "height": 800})
#         # await page.add_init_script(path="./node_modules/echarts/dist/echarts.min.js")
#         # print("add_init_script ok")
#         # 加载 ECharts 示例代码
#         await page.set_content(code)
#         # 等待图表渲染完成
#         # WARNING:使用选择器等待图表渲染完成，若模板修改的话，注意#chart是否为图表挂载的dom元素id
#         # self.page.wait_for_selector('#chart')
#         await page.wait_for_timeout(3000)
#         # 保存为图片
#         await page.screenshot(path=save_path)
#         await page.close()
