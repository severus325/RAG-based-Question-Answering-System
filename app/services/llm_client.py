from __future__ import annotations

from openai import OpenAI


SYSTEM_PROMPT = """你是一个严谨的企业知识库问答助手。
你会优先使用提供的参考资料回答问题，不要编造资料中没有的信息。
如果资料不足以支撑答案，要明确说明“当前资料不足”并指出还需要什么文档。
回答使用中文，尽量简洁，并在结尾附上参考来源文件名。"""


class QwenClient:
    def __init__(self, api_url: str, api_key: str, model: str, max_context_chars: int, max_tokens: int) -> None:
        self.api_url = api_url.strip()
        self.api_key = api_key.strip()
        self.model = model
        self.max_context_chars = max_context_chars
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=self.api_url, api_key=self.api_key) if self.api_url and self.api_key else None

    def answer(self, question: str, retrieved_chunks: list[dict[str, object]]) -> str:
        if not self.api_url:
            raise ValueError("未配置 QWEN_API_URL，请先在 .env 中填写。")
        if not self.api_key:
            raise ValueError("未配置 QWEN_API_KEY，请先在 .env 中填写。")
        if self.client is None:
            self.client = OpenAI(base_url=self.api_url, api_key=self.api_key)

        if retrieved_chunks:
            context_blocks: list[str] = []
            current_size = 0
            for index, chunk in enumerate(retrieved_chunks, start=1):
                block = f"[参考{index}] 来源: {chunk['source']}\n内容:\n{chunk['text']}\n"
                if current_size + len(block) > self.max_context_chars:
                    break
                context_blocks.append(block)
                current_size += len(block)
            context_text = "\n".join(context_blocks)
        else:
            context_text = "当前没有检索到任何参考资料。"

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"用户问题:\n{question}\n\n参考资料:\n{context_text}",
                },
            ],
        )
        content = response.choices[0].message.content
        return content.strip() if isinstance(content, str) else str(content).strip()
