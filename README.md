\# PDF‑Document‑Agent

基于LangGraph实现的文档RAG‑Agent，本地开源大模型PDF问答系统。



\## 技术栈

Python

LangGraph：构建Agent状态节点、会话记忆

LangChain‑Ollama：本地大模型调用

Chroma：向量数据库

nomic‑embed‑text：文本嵌入模型

PyPDFLoader：PDF文档解析

Gradio：网页交互UI



\## 功能特性

1\. 支持PDF文档上传，自动文本切分，构建向量知识库

2\. 基于LangGraph MemorySaver实现多轮会话记忆，thread\_id隔离会话

3\. 本地部署Qwen2‑7B大模型，基于文档内容回答用户问题

4\. Web网页端交互



\## 运行步骤

1.安装Ollama，拉取模型

```bash

ollama pull qwen2:7b

ollama pull nomic‑embed‑text

```

2\. 克隆项目，创建虚拟环境

```bash

git clone https://github.com/你的用户名/AI-agent-demo.git

cd AI-agent-demo

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

```

3\. 启动项目

```bash

python agent\_rag.py

```

