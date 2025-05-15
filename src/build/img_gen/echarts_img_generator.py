# from playwright.async_api import async_playwright
from http.server import SimpleHTTPRequestHandler
import os
from pathlib import Path
import re
import socket
import socketserver
import threading
from playwright.sync_api import sync_playwright

from src.build.code_gen.html_template import HtmlTemplate

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
        def check_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        while check_port_in_use(self.port):
            self.port += 1
        self.httpd = MyServer(("", self.port), QuietServer)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.start()
        print(f"EchartsImgGenerator:服务器已启动，监听端口 {self.port}")
        self.html_template = HtmlTemplate()

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

    def generate_img(self, code: str, save_path: str):
        """借助无头浏览器渲染 ECharts 图表并保存为图片"""
        if not re.search(r"<html>", code):
            # 如果不存在html标签，说明是js代码，需要使用模板包裹起来
            code = self.html_template.instance(code)
        page = self.browser.new_page(viewport={"width": 1920, "height": 1080})
        content = re.sub(
            r'src="[^"]*"',
            f'src="http://localhost:{self.port}/lib/echarts.min.js"',
            code,
        )
        page.set_content(content, wait_until="load")
        pattern = r'getElementById\((["\'])(.*?)\1\)'
        match = re.search(pattern, code)
        id = match.group(2)
        page.wait_for_selector(f"#{id}", state="visible")
        page.wait_for_timeout(3000)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        element = page.query_selector(f"id={id}")
        element.screenshot(path=save_path)
        page.close()
