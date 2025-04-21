from src.datamodel.annotation import Message


class BaseToEvalLLM:
    """定义待评估大模型的基类"""

    def __init__(self):
        pass

    def answer(self, messages: list[Message]):
        """根据消息列表生成回答
        需查看Message类型定义，并根据具体情况实现
        """
        pass
