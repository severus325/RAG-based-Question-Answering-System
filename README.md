# Qwen API RAG 系统

这个目录现在是一套直接调用 Qwen API 的 RAG 应用：

- 后端：`FastAPI`
- 前端：单页静态页面
- 模型调用：Qwen API
- 检索：本地 embedding + FAISS 向量召回 + 本地 rerank 重排

## 你需要填写的参数

在 `.env` 中填写下面两个参数：

```env
QWEN_API_URL=
QWEN_API_KEY=
QWEN_MODEL=qwen3.5-flash
MODEL_PYTHON_BIN=
ENABLE_RAG=true
EMBEDDING_MODEL_PATH=
RERANK_MODEL_PATH=
```

参数名就是：

- `QWEN_API_URL`
- `QWEN_API_KEY`

模型当前默认是：

- `QWEN_MODEL=qwen3.5-flash`

本地检索模型当前使用：

- `EMBEDDING_MODEL_PATH=bge-m3`
- `RERANK_MODEL_PATH=model/bge-reranker-v2-m3`

## 目录结构

```text
app/
  core/config.py
  services/neural_worker.py
  services/document_loader.py
  services/retrieval.py
  services/llm_client.py
  templates/index.html
  static/
data/documents/
data/index/index.json
scripts/retrieval_model_worker.py
```

## 放文档

把基础文档放进：

```text
data/documents/
```

支持格式：

- `.md`
- `.txt`
- `.pdf`
- `.docx`

## 启动方式

启动 Web 服务：

```bash
bash scripts/start_rag_app.sh
```

浏览器打开：

```text
http://127.0.0.1:7861
```

## 页面功能

- 显示当前 RAG 是否开启，并支持前端开关
- 查看当前知识库索引状态
- 上传文档并自动重建索引
- 删除单个文档并自动重建索引
- 手动重建索引
- 输入问题并发起 RAG 问答
- 返回答案和命中的参考片段

## 当前实现说明

- 检索层现在是两阶段：
- 第一阶段用本地 `bge-m3` 生成 embedding，并写入 `FAISS` 索引做候选召回
- 第二阶段用本地 `bge-reranker-v2-m3` 对候选片段做 rerank
- 生成层直接调用 Qwen API，不再依赖本地模型服务
- 你填好 `QWEN_API_URL` 和 `QWEN_API_KEY` 后就可以直接使用
- Web 进程默认跑在 `APP_PYTHON_BIN`
- embedding/rerank 模型 worker 默认跑在 `MODEL_PYTHON_BIN`
