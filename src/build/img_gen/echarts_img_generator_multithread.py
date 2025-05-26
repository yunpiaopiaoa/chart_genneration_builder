import html
from http.server import SimpleHTTPRequestHandler
import logging
import os
import re
import socket
import socketserver
import threading
import playwright
from playwright.sync_api import sync_playwright
from playwright.sync_api._generated import Browser

from src.build.code_gen.html_template import HtmlTemplate
from .base_img_generator import BaseImgGenerator


class QuietServer(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


class MyServer(socketserver.TCPServer):
    allow_reuse_address = True  # 允许程序重新运行后重新绑定端口


class EchartsImgGeneratorMultiThread(BaseImgGenerator):
    """只使用同步"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"log/{__name__}.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    thread_local = threading.local()

    def __init__(self):
        self.port = 8000
        self.delay=1000#毫秒
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

    def get_thread_browser(self) -> Browser:
        if not hasattr(self.thread_local, 'playwright'):
            self.thread_local.playwright = sync_playwright().start()
            self.thread_local.browser = self.thread_local.playwright.chromium.launch()
        return self.thread_local.browser

    def generate_img(self, code: str, save_path: str):
        """借助无头浏览器渲染 ECharts 图表并保存为图片"""
        browser=self.get_thread_browser()
        if not re.search(r"html", code):
            # 如果不存在html标签，说明是js代码，需要使用模板包裹起来
            code = self.html_template.instance(code)
        code = html.unescape(code)# 还原转义字符
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        content = re.sub(
            r'src=(["\'])([^"\']*)\1',
            f'src=\\1http://localhost:{self.port}/lib/echarts.min.js\\1',
            code,
        )
        # self.logger.info(content)
        page.set_content(content, wait_until="load")
        pattern = r'getElementById\((["\'])(.*?)\1\)'
        match = re.search(pattern, code)
        id = match.group(2)
        page.wait_for_selector(f"#{id}", state="visible")
        page.wait_for_timeout(self.delay)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        element = page.query_selector(f"id={id}")
        element.screenshot(path=save_path)
        page.close()

    def stop_server(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
            print(f"EchartsImgGenerator:服务器已关闭")
        if self.server_thread:
            self.server_thread.join()
            print(f"EchartsImgGenerator:服务器线程已结束")

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        # WARNING:访问不到浏览器资源，不确定是否释放
        for thread in threading._active.values():
            # print("release",thread.name)
            if hasattr(thread, 'thread_local'):
                print(hasattr(thread.thread_local, 'browser'),hasattr(thread.thread_local, 'playwright'))
                if hasattr(thread.thread_local, 'browser'):
                    thread.thread_local.browser.close()
                    del thread.thread_local.browser
                if hasattr(thread.thread_local, 'playwright'):
                    thread.thread_local.playwright.stop()
                    del thread.thread_local.playwright
        self.stop_server()
