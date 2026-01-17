from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import BaseMessage

from typing import List, Dict, Any

from fastapi import HTTPException, FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

import os

from utils import ImageProcessor, AudioProcessor, PDFProcessor
import re
from datetime import datetime
import json
from dotenv import load_dotenv

from rag_manager import RAGManager
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # 从 .env 文件加载环境变量

# 模型初始化
def get_chat_model():
    try:
        model = init_chat_model(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",  
            model_provider="openai",  
            base_url="https://api.siliconflow.cn/v1/",  
            api_key=os.getenv("GUIJI_API_KEY"),  
        )
        return model
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {str(e)}")
    

#------------------------------------------------------------
# 数据结构定义
class ContentBlock(BaseModel): # 内容块定义
    type: str = Field(..., description="内容块类型，如'text', 'image', 'audio'")
    content: str = Field(..., description="内容块的实际内容，文本或URL等")

class MessageRequest(BaseModel): # 请求格式定义
    content_blocks : List[ContentBlock] = Field(default=[], description="消息内容块列表")
    history : List[Dict[str, Any]] = Field(default=[], description="历史消息列表")
    pdf_chunks : List[Dict[str, Any]] = Field(default=[], description="PDF内容块列表")

class MessageResponse(BaseModel): # 响应格式定义
    content : str
    timestamp : str
    role : str
    reference: List[Dict[str, Any]]


#------------------------------------------------------------
# 多模态message构建
def create_multimodal_message(request: MessageRequest, image_file: UploadFile | None, audio_file: UploadFile | None):
    # 处理图片内容
    messages = []
    request_text_content = ""
    if image_file:
        processor = ImageProcessor()
        image_base64 = processor.image_to_base64(image_file)
        mime_type = processor.get_image_mime_type(image_file.filename)
        messages.append({
            "type" : "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        })
    # 处理音频内容
    if audio_file:
        processor = AudioProcessor()
        audio_base64 = processor.audio_to_base64(audio_file)
        mime_type = processor.get_audio_mime_type(audio_file.filename)
        messages.append({
            "type" : "audio_url",
            "audio_url": {
                "url": f"data:{mime_type};base64,{audio_base64}"
            }
        })
    # 处理文本内容块
    for i, block in enumerate(request.content_blocks):
        if block.type == "text":
            messages.append({
                "type": "text",
                "text": block.content
            })
            request_text_content += block.content + " "
        elif block.type == "image":
            if block.content.startswith("data:image/"):  # base64图像
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": block.content
                    }
                })
        elif block.type == "audio":
            if block.content.startswith("data:audio/"):  # base64音频
                messages.append({
                    "type": "audio_url",
                    "audio_url": {
                        "url": block.content
                    }
                })
        
        if request.pdf_chunks:
            pdf_content = "\n\n=== 参考文档内容 ===\n"
            for i, chunk in enumerate(request.pdf_chunks):
                source_info = chunk.get("metadata", {}).get("source_info", f"文档片段 {i+1}")
                pdf_content += f"\n[{i+1}]\n{chunk.get('content', '')}\n[来源: {source_info}\n]"
            pdf_content += "\n=== 参考文档内容结束 ===\n"
            pdf_content += "\n请在回答时引用相关内容，使用格式如 [1]、[2] 等。\n"

            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["type"] == "text":
                    messages[i]["text"] += pdf_content
                    break
        
        # 1. 基于request的内容检索相关chunks
        rag_manager = RAGManager()
        relevant_chunks = rag_manager.retrieve_relevant_chunks(request_text_content, top_k=5)
        # 2. 将检索到的内容添加到messages中
        if not relevant_chunks:
            pass
        else:
            context = "参考文档内容：\n"
            for idx, chunk in enumerate(relevant_chunks, 1):
                doc_title = chunk.get("document_title", "未知")
                content = chunk.get("content", "")
                similarity = chunk.get("score", 0)
                vec_sim = chunk.get("vector_similarity")
                txt_sim = chunk.get("text_similarity")
                page = f"(第{chunk.get('page_number')}页)" if chunk.get('page_number') else ""
                
                context += f"\n[{idx}] 来源: {doc_title} {page}\n相似度: {similarity}, 向量相似度: {vec_sim}, 文本相似度: {txt_sim}\n内容: {content}\n"
            messages.append({
                "type": "text",
                "text": context
            })
            
    return HumanMessage(content=messages)
    
#------------------------------------------------------------
# history管理
def convert_history_to_messages(history: List[Dict[str, Any]] ) -> List[BaseMessage]:
    # 将历史记录转换为消息列表
    messages = []

    # 添加系统消息
    system_prompt = """
        你是一个专业的多模态 RAG 助手，具备如下能：
        1. 与用户对话的能力。
        2. 图像内容识别和分析能力(OCR, 对象检测， 场景理解)
        3. 音频转写与分析
        4. 知识检索与问答
        
        重要指导原则：
        - 当用户上传图片并提出问题时，请结合图片内容和用户的具体问题来回答
        - 仔细分析图片中的文字、图表、对象、场景等所有可见信息
        - 根据用户的问题重点，有针对性地分析图片相关部分
        - 如果图片包含文字，请准确识别并在回答中引用
        - 如果用户只上传图片没有问题，则提供图片的全面分析
        
        引用格式要求（重要）：
        - 当回答基于提供的参考文档内容时，必须在相关信息后添加引用标记，格式为[1]、[2]等
        - 引用标记应紧跟在相关内容后面，如："这是重要信息[1]"
        - 每个不同的文档块使用对应的引用编号
        - 如果用户消息中包含"=== 参考文档内容 ==="部分，必须使用其中的内容来回答问题并添加引用
        - 只需要在正文中使用角标引用，不需要在最后列出"参考来源"
        - 如果没有提供参考文档，请不要添加任何引用标记，更不要编造引用
        
        请以专业、准确、友好的方式回答，并严格遵循引用格式。当有参考文档时，优先使用文档内容回答。
    """

    messages.append(SystemMessage(content=system_prompt))

    for msg in history:
        role = msg.get("role", "user")
        content_blocks = msg.get("content_blocks", [])
        msg_content = []
        if role == "user":
            for block in content_blocks:
                if block["type"] == "text":
                    msg_content.append({
                        "type": "text",
                        "text": block["content"]
                    })
                elif block["type"] == "image":
                    if block["content"].startswith("data:image/"):
                        msg_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": block["content"]
                            }
                        })
                elif block["type"] == "audio":
                    if block["content"].startswith("data:audio/"):
                        msg_content.append({
                            "type": "audio_url",
                            "audio_url": {
                                "url": block["content"]
                            }
                        })
            messages.append(HumanMessage(content=msg_content))
        elif role == "assistant":
            messages.append(AIMessage(content=msg.get("content", "")))
    return messages

#------------------------------------------------------------
# 引用提取工具
def extract_references_from_response(content:str, pdf_chunks:list=None) :
    # 从回答内容中提取引用，并返回对应的文档chunk
    if not pdf_chunks:
        return []
    references = []
    reference_pattern = r'[(\d+)]'
    matches = re.findall(reference_pattern, content)
    if matches:
        for match in matches:
            ref_num = int(match)-1
            if 0 <= ref_num < len(pdf_chunks):
                references.append({
                    "id": pdf_chunks[ref_num].get("id", ""),
                    "content": pdf_chunks[ref_num].get("content", "")[:200] + "..." if len(pdf_chunks[ref_num].get("content", "")) > 200 else pdf_chunks[ref_num].get("content", ""),
                    "metadata": pdf_chunks[ref_num].get("metadata", {})
                })
    
    return references

def get_hisotry_from_db(user_id: str, session_id: str, limit: int =5) -> List[Dict[str, Any]]:
    from db_manager import DatabaseManager
    db_manager = DatabaseManager()
    conversation_id = db_manager.get_conversation_id(user_id=user_id, session_id=session_id)
    history_data = db_manager.get_recent_messages(conversation_id=conversation_id, limit=limit)
    # 将DB消息转换为历史格式：
    # - user: 使用单文本块表示
    # - assistant: 直接使用content字段
    history = []
    for his in history_data:
        role = his.get("role")
        content = his.get("content", "")
        if role == "user":
            history.append({
                "role": "user",
                "content_blocks": [{"type": "text", "content": content}]
            })
        elif role == "assistant":
            history.append({
                "role": "assistant",
                "content": content
            })
        else:
            # 兜底：未知角色按文本块处理
            history.append({
                "role": role or "user",
                "content_blocks": [{"type": "text", "content": content}]
            })
    return history

def add_history_to_db(user_id: str, session_id: str, role: str,  content: str, limit: int = 5):
    from db_manager import DatabaseManager
    db_manager = DatabaseManager()
    conversation_id = db_manager.get_conversation_id(user_id=user_id, session_id=session_id)
    msg_id = db_manager.add_message_with_limit(conversation_id=conversation_id, role=role, content=content, limit=limit)
    return msg_id
#------------------------------------------------------------
# 流式输出响应
async def generate_streaming_response(messages: List[BaseMessage], pdf_chunks: List[Dict[str, Any]] = None,
                                      user_id: str | None = None, session_id: str | None = None,
                                      user_msg_text: str | None = None):
    chat_model = get_chat_model()
    try:
        full_response = ""
        chunk_count = 0

        async for chunk in chat_model.astream(messages):
            chunk_count += 1
            full_response += chunk.content

            data = {
                "type" : "content_delta",
                "content" : chunk.content,
                "timestamp" : datetime.utcnow().isoformat() + "Z",
            }
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        # 最终响应，包含引用信息
        references = extract_references_from_response(full_response, pdf_chunks)
        final_data = {
            "type" : "final_response",
            "content" : full_response,
            "timestamp" : datetime.utcnow().isoformat() + "Z",
            "reference" : references
        }
        # 同步持久化：与 chat_sync 保持一致逻辑（写入 user 与 assistant）
        if user_id and session_id:
            try:
                if user_msg_text:
                    add_history_to_db(user_id=user_id, session_id=session_id, role="user", content=user_msg_text, limit=5)
                add_history_to_db(user_id=user_id, session_id=session_id, role="assistant", content=full_response, limit=5)
            except Exception as persist_err:
                logger.error(f"持久化历史失败: {persist_err}")
        yield f"data: {json.dumps(final_data, ensure_ascii=False)}\n\n"
    except Exception as e:
        error_data = {
            "type" : "error",
            "message" : f"响应生成失败: {str(e)}",
            "timestamp" : datetime.utcnow().isoformat() + "Z",
        }
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

#------------------------------------------------------------
# 同步响应（便于本地调试）
def generate_response_sync(messages: List[BaseMessage], pdf_chunks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    chat_model = get_chat_model()
    try:
        result = chat_model.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        references = extract_references_from_response(content, pdf_chunks)
        return {
            "type": "final_response",
            "content": content,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "reference": references
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"响应生成失败: {str(e)}")

#------------------------------------------------------------
# FastAPI应用初始化
app = FastAPI(
    title  = "多模态 RAG 助手 API",
    description = "基于 LangChain 和 Qwen-Omni 的多模态 RAG 助手服务",
    version = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/chat/stream")
async def chat_stream(
    content_blocks : str = Form(default="[]"),
    user_id : str = Form(default="test_user"),
    session_id : str = Form(default="test_session"),
    image_file: UploadFile | None = File(default=None),
    audio_file: UploadFile | None = File(default=None),
    pdf_file : UploadFile | None = File(default=None)
):
    try:
        try:
            content_blocks_data = json.loads(content_blocks)
            # 从数据库读取最近历史（最多5条），与 chat_sync 一致
            history_data = get_hisotry_from_db(user_id=user_id, session_id=session_id, limit=5)
            logger.info(f"content_blocks_data: {content_blocks_data}")
            logger.info(f"history_data: {history_data}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"json解析失败: {str(e)}")
        
        if pdf_file:
            pdf_processor = PDFProcessor()
            pdf_content = await pdf_file.read()
            pdf_chunks_data = await pdf_processor.process_pdf(pdf_content, pdf_file.filename)
            request_data = MessageRequest(
                content_blocks=content_blocks_data,
                history=history_data,
                pdf_chunks=pdf_chunks_data
            )
        else:
            request_data = MessageRequest(
                content_blocks=content_blocks_data,
                history=history_data
            )
        
        messages = convert_history_to_messages(request_data.history)
        user_message = create_multimodal_message(request_data, image_file, audio_file)
        logger.info(f"用户消息列表: {user_message}")
        messages.append(user_message)

        # 聚合用户文本用于持久化（与 chat_sync 相同）
        msg_to_store = ""
        for block in request_data.content_blocks:
            if block.type == "text":
                msg_to_store += block.content + " "
            elif block.type == "image":
                pass
            elif block.type == "audio":
                pass

        return StreamingResponse(
            generate_streaming_response(messages, request_data.pdf_chunks, user_id=user_id, session_id=session_id, user_msg_text=msg_to_store), # 异步生成器，含持久化
            media_type="text/event-stream", # 媒体类型，固定搭配
            headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"请求处理失败: {str(e)}")

#------------------------------------------------------------
# 同步调试入口（非路由）：返回完整响应字典，便于在脚本/测试中直接调用
def chat_sync(
    content_blocks : str = "[]",
    user_id : str = "test_user",
    session_id : str = "test_session",
    image_file: UploadFile | None = None,
    audio_file: UploadFile | None = None,
    pdf_file : UploadFile | None = None
):
    try:
        try:
            content_blocks_data = json.loads(content_blocks)
            history_data = get_hisotry_from_db(user_id=user_id, session_id=session_id, limit=5)
            logger.info(f"content_blocks_data: {content_blocks_data}")
            logger.info(f"history_data: {history_data}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"json解析失败: {str(e)}")
        
        if pdf_file:
            pdf_processor = PDFProcessor()
            pdf_content = pdf_file.read()
            pdf_chunks_data = pdf_processor.process_pdf(pdf_content, pdf_file.filename)
            request_data = MessageRequest(
                content_blocks=content_blocks_data,
                history=history_data,
                pdf_chunks=pdf_chunks_data
            )
        else:
            request_data = MessageRequest(
                content_blocks=content_blocks_data,
                history=history_data
            )
        
        messages = convert_history_to_messages(request_data.history)
        user_message = create_multimodal_message(request_data, image_file, audio_file)
        messages.append(user_message)
        logger.info(f"用户消息列表: {user_message}")

        response = generate_response_sync(messages, request_data.pdf_chunks)

        msg_to_store = ""
        for block in request_data.content_blocks:
            if block.type == "text":
                msg_to_store += block.content + " "
            elif block.type == "image":
                pass
            elif block.type == "audio":
                pass
        add_history_to_db(user_id=user_id, session_id=session_id, role="user", content=msg_to_store, limit=5)
        add_history_to_db(user_id=user_id, session_id=session_id, role="assistant", content=response["content"], limit=5)
        # 直接返回一次性完整响应（非流式）
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"请求处理失败: {str(e)}")

#------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

    


