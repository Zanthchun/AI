import os

# ⚠️ 设置环境变量，确保重排模型能以几 MB/s 的速度在国内极速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import chromadb
from openai import OpenAI
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions

# ⚠️ 引入轻量级 PDF 解析库
import fitz

# ⚠️ 使用更标准的句子交叉编码器进行重排
from sentence_transformers import CrossEncoder

# ❌ 原来的不安全写法：
# API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx" 
#  现在的安全写法：
# 让代码去云端服务器的环境变量里秘密读取一个叫 "DEEPSEEK_API_KEY" 的东西
import os
API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

print("📁 正在通过国内镜像加载 BGE 向量模型 (首次会自动全速下载)...")
zh_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5",
    device="cpu"
)

print("⚙️ 正在加载 BGE 重排引擎...")
ranker = CrossEncoder("BAAI/bge-reranker-base")

COMPANY_KEYWORDS = ["比亚迪", "长安", "宁德时代", "广汽", "吉利", "长城", "特斯拉",
                    "理想", "蔚来", "小鹏", "赛力斯"]

# 🌟 全局历史对话存储字典
GLOBAL_CHAT_HISTORY = {}


# --- 1.5 高亮渲染函数 ---
def format_source_html(text):
    if not text or text == "等待检索...":
        return "<div style='border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; height: 550px; display: flex; justify-content: center; align-items: center; color: gray;'>等待检索...</div>"

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'(\d+\.?\d*%|\d+\.?\d*亿|\d+\.?\d*万)',
                  r'<span style="color:#e63946;font-weight:bold;background-color:#ffe3e3;padding:0 2px;border-radius:2px;">\1</span>',
                  text)
    text = re.sub(r'(\[来源:.*?\])',
                  r'<span style="color:#1d4ed8;font-weight:bold;">\1</span>', text)
    text = text.replace("\n", "<br>")

    html_template = f"""
    <div style='border: 1px solid #e5e7eb; border-radius: 8px; padding: 15px; background-color: #f9fafb; height: 550px; overflow-y: auto; box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.03);'>
        <h3 style='margin-top: 0; color: #374151; font-size: 16px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; position: sticky; top: 0; background-color: #f9fafb;'>📊 数据溯源面板 (经隔离与精排)</h3>
        <div style='font-size:14px; line-height:1.8; color: #4b5563; margin-top: 10px;'>
            {text}
        </div>
    </div>
    """
    return html_template


# --- 2. 知识库加载函数 (纯净 PyMuPDF 版) ---
def init_db():
    chroma_client = chromadb.PersistentClient(path="./my_vector_db")
    collection = chroma_client.get_or_create_collection(name="auto_reports", embedding_function=zh_ef)

    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return collection

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return collection

    for filename in pdf_files:
        path = os.path.join(data_dir, filename)

        # 避免重复解析
        try:
            if collection.get(ids=[f"{filename}_0"])['ids']:
                continue
        except Exception:
            pass

        company_tag = "未知"
        for k in COMPANY_KEYWORDS:
            if k in filename:
                company_tag = k
                break
        if "政策" in filename or "宏观" in filename or "规划" in filename:
            company_tag = "宏观政策"

        print(f"📄 正在使用 PyMuPDF 解析: {filename} -> 标签: 【{company_tag}】")
        try:
            doc = fitz.open(path)
            text = ""
            for page in doc:
                page_text = page.get_text()
                # 简单清洗多余的空白符，但绝不触碰中文字符或数字
                page_text = re.sub(r'\s+', ' ', page_text)
                text += page_text + "\n"
            doc.close()

            # ⚠️ 删除了之前那个会导致中文字符全灭的 cid 剔骨正则！

            if not text.strip():
                print(f"⚠️ 警告：{filename} 提取文本为空，可能是图片型 PDF 或极端加密文件！建议更换为券商研报。")
                continue

            chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_text(text)

            collection.add(
                documents=[f"[来源: {filename}]\n{c}" for c in chunks],
                ids=[f"{filename}_{i}" for i in range(len(chunks))],
                metadatas=[{"company": company_tag} for _ in range(len(chunks))]
            )
            print(f"✅ 成功入库: {filename} ({len(chunks)} 个数据块)")
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")

    return collection


# --- 3. 核心交互逻辑与会话管理 ---

def create_new_chat():
    return [], "", gr.update(value=None), format_source_html("等待检索...")


def switch_chat_session(selected_session):
    if selected_session and selected_session in GLOBAL_CHAT_HISTORY:
        loaded_history = GLOBAL_CHAT_HISTORY[selected_session]
        return loaded_history, selected_session, format_source_html("✅ 已加载历史对话记录...")
    return [], "", format_source_html("等待检索...")


def user_submit(user_message, history, current_session):
    if history is None:
        history = []
    history.append({"role": "user", "content": user_message})

    if not current_session:
        current_session = user_message[:12] + ("..." if len(user_message) > 12 else "")

    GLOBAL_CHAT_HISTORY[current_session] = history
    session_choices = list(GLOBAL_CHAT_HISTORY.keys())

    return "", history, current_session, gr.update(choices=session_choices, value=current_session)


def bot_response(history, style_mode, current_session):
    if not history or history[-1]["role"] != "user":
        yield history, format_source_html(""), current_session
        return

    raw_content = history[-1]["content"]
    user_message = raw_content[0]["text"] if isinstance(raw_content, list) else raw_content

    try:
        collection = chromadb.PersistentClient(path="./my_vector_db").get_collection(
            name="auto_reports",
            embedding_function=zh_ef
        )

        # 🛠️ 提取所有提及的公司，支持多实体对比
        target_companies = [k for k in COMPANY_KEYWORDS if k in user_message]

        if target_companies:
            # 构建复合过滤字典
            filter_list = [{"company": comp} for comp in target_companies]
            filter_list.append({"company": "宏观政策"})
            search_filter = {"$or": filter_list}

            # 按比例扩大初排召回深度
            base_n_results = 40 * len(target_companies)
            results = collection.query(query_texts=[user_message], n_results=base_n_results, where=search_filter)
        else:
            results = collection.query(query_texts=[user_message], n_results=40)

        if not results['documents'] or len(results['documents'][0]) == 0:
            context = "⚠️ 知识库未检索到相关内容。"
            formatted_html = format_source_html("等待检索...")
        else:
            retrieved_docs = results['documents'][0]

            # 1. 用户问题与粗排文档两两组合
            pairs = [[user_message, doc] for doc in retrieved_docs]

            # 2. 扔进 BGE 重排模型计算精准得分
            scores = ranker.predict(pairs)

            # 3. 将文档与得分打包，按得分从高到低排序
            reranked_results = sorted(
                zip(retrieved_docs, scores),
                key=lambda x: x[1],
                reverse=True
            )

            # 4. 动态扩大精排后的 Top-N
            top_k = 8 + (max(0, len(target_companies) - 1) * 6)
            top_n_results = [doc for doc, score in reranked_results[:top_k]]

            context = "\n\n".join(top_n_results)
            formatted_html = format_source_html(context)

    except Exception as e:
        context = f"⚠️ 数据库检索失败。\n报错: {e}"
        formatted_html = format_source_html(f"错误信息: {e}")

    if style_mode == "专家模式":
        style_instruction = """
        👉 当前语言风格模式：【专家模式】
        - 动作：使用极其专业、严谨、深度的金融分析师语言进行回复。
        - 要求：多使用专业术语，逻辑推演深。必须使用【Markdown表格】呈现核心定量数据，并进行深度的【定量数据归因】。
        """
    else:
        style_instruction = """
        👉 当前语言风格模式：【人话模式】
        - 动作：使用最通俗易懂、接地气、没有门槛的‘大白话’来解释。
        - 要求：【绝对禁止】堆砌金融术语。若必须提到术语，必须用生活常识、生动幽默的比喻来翻译，让外行人也能秒懂。
        """

    system_prompt = f"""你是一位多才多艺的资深投研专家。你的任务是基于提供的参考资料，提供高度精准的解答。

        🚨 【核心风格约束（必须严格遵守）】 🚨
        {style_instruction}

        【专业底线约束】
        1. 📌 结论先行：开篇必须一句话给出核心结论。
        2. 🔍 严格溯源：所有核心财务数据必须明确标出其来源文档（例如：[来源: 某某年报.pdf]）。
        3. 🛡️ 绝对防幻觉：严格核对提问主体！资料里没有的数据坦诚回答暂无信息，绝不捏造。如果是对比问题，请分别清晰列出各家车企的数据，切勿张冠李戴。

        ---
        参考资料：
        {context}"""

    messages = [{"role": "system", "content": system_prompt}]
    recent_history = history[-6:] if len(history) > 6 else history

    for msg in recent_history:
        msg_content = msg["content"]
        if isinstance(msg_content, list):
            msg_content = msg_content[0]["text"]
        messages.append({"role": msg["role"], "content": msg_content})

    history.append({"role": "assistant", "content": ""})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            temperature=0.3 if style_mode == "人话模式" else 0.1
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                history[-1]["content"] += chunk.choices[0].delta.content

                GLOBAL_CHAT_HISTORY[current_session] = history
                yield history, formatted_html, current_session
    except Exception as e:
        history[-1]["content"] = f"⚠️ API 调用失败。\n报错: {e}"
        yield history, formatted_html, current_session


# --- 4. 界面构建 ---
with gr.Blocks(title="新能源汽车 AI 投研助手", theme=gr.themes.Soft()) as demo:
    current_session = gr.State("")

    gr.Markdown("## 📈 新能源汽车 AI 投研助手 (深度重排版)")

    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("### 🗂️ 会话管理")
            new_chat_btn = gr.Button("➕ 新建对话", variant="primary")

            gr.Markdown("---")
            history_list = gr.Radio(
                choices=[],
                label="🕒 历史聊天记录",
                interactive=True
            )

        with gr.Column(scale=3):
            mode_selector = gr.Radio(
                choices=["专家模式", "人话模式"],
                value="专家模式",
                label="🧠 语言风格滤镜 (可随时切换模式并重新提问)",
                interactive=True
            )
            chatbot = gr.Chatbot(label="投研对话窗口", height=520)

            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="请输入投研问题，例如：对比比亚迪和理想的单车毛利率...",
                    scale=4
                )
                submit_btn = gr.Button("发送", variant="primary", scale=1)

        with gr.Column(scale=1):
            source_box = gr.HTML(value=format_source_html("等待检索..."))

    # --- 🌟 事件绑定逻辑 ---
    new_chat_btn.click(
        create_new_chat,
        inputs=[],
        outputs=[chatbot, current_session, history_list, source_box]
    )

    history_list.change(
        switch_chat_session,
        inputs=[history_list],
        outputs=[chatbot, current_session, source_box]
    )

    msg.submit(
        user_submit,
        inputs=[msg, chatbot, current_session],
        outputs=[msg, chatbot, current_session, history_list],
        queue=False
    ).then(
        bot_response,
        inputs=[chatbot, mode_selector, current_session],
        outputs=[chatbot, source_box, current_session]
    )

    submit_btn.click(
        user_submit,
        inputs=[msg, chatbot, current_session],
        outputs=[msg, chatbot, current_session, history_list],
        queue=False
    ).then(
        bot_response,
        inputs=[chatbot, mode_selector, current_session],
        outputs=[chatbot, source_box, current_session]
    )

if __name__ == "__main__":
    print("正在初始化融合版知识库...")
    init_db()
    print("系统启动成功！将自动在浏览器中打开新窗口...")

    demo.launch(
        inbrowser=True,
        auth=("sysu", "123456"),
        share=False
    )
