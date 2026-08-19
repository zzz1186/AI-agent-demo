from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Annotated, Sequence
import operator
import gradio as gr

llm = ChatOllama(model="qwen2:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

PERSIST_DIR = "./chroma_db"
vector_store = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)


class AgentState(dict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: str


def retrieve_node(state: AgentState):
    latest_query = state["messages"][-1].content
    docs = vector_store.similarity_search(latest_query, k=3)
    docs_text = "\n".join([d.page_content for d in docs])
    return {"context": docs_text}


def generate_node(state: AgentState):
    prompt_template = """基于下面文档上下文回答用户问题。
如果文档没有相关信息，直接说明文档未包含该内容，不要编造。

【文档上下文】
{context}

用户问题：{question}
"""
    question = state["messages"][-1].content
    prompt = prompt_template.format(context=state["context"], question=question)
    resp = llm.invoke(prompt)
    return {"messages": [resp]}


builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge("retrieve", "generate")
builder.set_entry_point("retrieve")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 关键改动：完全不使用gradio history组装消息，交给langgraph memory管理
def chat_handler(message, history):
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
    return result["messages"][-1].content


def upload_pdf(file_obj):
    if not file_obj:
        return "请选择PDF文件"
    loader = PyPDFLoader(file_obj.name)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    chunks = splitter.split_documents(pages)
    vector_store.add_documents(chunks)
    return f"PDF解析完成，共切分 {len(chunks)} 个文本块，可以开始提问。"


with gr.Blocks(title="PDF文档AI Agent") as demo:
    gr.Markdown("# 📄 PDF文档问答Agent｜LangGraph+Ollama")
    pdf_file = gr.File(label="上传PDF文档", file_types=[".pdf"])
    upload_info = gr.Textbox(label="上传状态", interactive=False)
    pdf_file.upload(upload_pdf, inputs=pdf_file, outputs=upload_info)
    gr.ChatInterface(fn=chat_handler, title="文档对话", description="上传PDF后，向文档提问")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
