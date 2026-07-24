import os
import json
import base64
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dotenv import load_dotenv

_ENV_LOADED = False
def _ensure_env():
    global _ENV_LOADED
    if not _ENV_LOADED:
        env_path = Path(__file__).parent.parent / "configs" / "keys.env"
        if env_path.exists():
            load_dotenv(str(env_path))
        _ENV_LOADED = True

_ensure_env()


class LLMClient(ABC):
    """多模态大模型调用公共接口"""

    def __init__(self):
        self.llm = None

    @abstractmethod
    def init_llm(self, **kwargs):
        ...

    @abstractmethod
    def invoke(self, messages: list, **kwargs) -> str:
        ...


class OllamaClient(LLMClient):
    def init_llm(self, model=None, base_url=None, temperature=0.1, **kwargs):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(
            base_url=base_url or os.getenv("OLLAMA_URL", "http://localhost:11434"),
            model=model or os.getenv("OLLAMA_MODEL", "qwen2.5-vl:7b"),
            temperature=temperature,
            **kwargs
        )
        return self

    def invoke(self, messages, **kwargs) -> str:
        resp = self.llm.invoke(messages)
        return resp.content


class TongyiClient(LLMClient):
    def init_llm(self, model=None, **kwargs):
        from langchain_community.chat_models import ChatTongyi
        self.llm = ChatTongyi(
            model=model or os.getenv("TONGYI_MODEL", "qwen-vl-plus"),
            **kwargs
        )
        return self

    def invoke(self, messages, **kwargs) -> str:
        resp = self.llm.invoke(messages)
        return resp.content


class QianfanClient(LLMClient):
    def init_llm(self, model=None, **kwargs):
        from langchain_community.chat_models import QianfanChatEndpoint
        self.llm = QianfanChatEndpoint(
            model=model or os.getenv("QIANFAN_MODEL", "ERNIE-4.5-8K"),
            **kwargs
        )
        return self

    def invoke(self, messages, **kwargs) -> str:
        resp = self.llm.invoke(messages)
        return resp.content


class SparkClient(LLMClient):
    def init_llm(self, model=None, **kwargs):
        from langchain_community.chat_models import ChatSparkLLM
        self.llm = ChatSparkLLM(
            model=model or os.getenv("SPARK_MODEL", "Spark4.0"),
            **kwargs
        )
        return self

    def invoke(self, messages, **kwargs) -> str:
        resp = self.llm.invoke(messages)
        return resp.content


_PROVIDER_MAP = {
    "ollama": OllamaClient,
    "tongyi": TongyiClient,
    "qianfan": QianfanClient,
    "spark": SparkClient,
}

def init_llm(provider: str = "ollama", **kwargs) -> LLMClient:
    """工厂函数：创建并初始化 LLM 客户端

    Args:
        provider: 模型提供商 (ollama/tongyi/qianfan/spark)
        **kwargs: 传递给 init_llm 的参数 (model, base_url, temperature 等)

    Returns:
        已初始化的 LLMClient 实例
    """
    cls = _PROVIDER_MAP.get(provider)
    if cls is None:
        raise ValueError(f"不支持的模型提供商: {provider}，可选: {list(_PROVIDER_MAP.keys())}")
    client = cls()
    client.init_llm(**kwargs)
    return client


# ----- 辅助函数 -----

def image_file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def image_data_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def make_multimodal_message(
    system_prompt: str,
    user_text: str,
    image_base64_list: Optional[List[str]] = None,
) -> list:
    """构造多模态消息列表 (LangChain 格式)

    Args:
        system_prompt: 系统提示词
        user_text: 用户文本
        image_base64_list: base64 编码的图像列表 (可选)

    Returns:
        [SystemMessage, HumanMessage] 消息列表
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    content = []
    if user_text:
        content.append({"type": "text", "text": user_text})
    if image_base64_list:
        for b64 in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    messages.append(HumanMessage(content=content))
    return messages


def load_sys_prompt(key: str = "label_suggestion") -> str:
    """从 data/sys_prompt.json 读取系统提示词"""
    path = Path(__file__).parent.parent / "data" / "sys_prompt.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        return prompts.get(key, "")
    return ""


def load_label_cn_map() -> Dict[str, str]:
    """从 data/label_cn.txt 加载中文标签映射"""
    path = Path(__file__).parent.parent / "data" / "label_cn.txt"
    mapping = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    en, cn = line.split(":", 1)
                    mapping[en.strip()] = cn.strip()
    return mapping


def get_label_cn(label: str) -> str:
    """获取标签的中文名称，若无映射则返回原标签"""
    mapping = load_label_cn_map()
    return mapping.get(label, label)
