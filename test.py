from main import chat_sync
import logging
import json
import requests


STREAM_TIMEOUT = 300  # seconds

def test_chat_interface(content_blocks, user_id, session_id, image_file=None, audio_file=None, pdf_file=None):
    try:
        url = "http://localhost:8000/api/chat/stream"
        files = None
        payload = {
            "content_blocks": json.dumps([content_blocks]),
            "user_id": user_id,
            "session_id": session_id,
        }
        if image_file:
            files = {
                "image_file": ("image.jpg", open(image_file, "rb"), "image/jpeg")
            }
        if audio_file:
            files = {
                "audio_file": ("audio.m4a", open(audio_file, "rb"), "audio/mp4")
            }
        if pdf_file:
            files = {
                "pdf_file": ("document.pdf", open(pdf_file, "rb"), "application/pdf")
            }
        response = requests.post(
        url=url,
        data=payload,  
        files=files if audio_file or image_file or pdf_file else None,
        stream=True,
        timeout=STREAM_TIMEOUT,
        )
        if response.status_code == 200:
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(b'data: '):
                    try:
                        obj = json.loads(line[6:].decode('utf-8'))
                        msg_type = obj.get("type")
                        if msg_type == "content_delta":
                            print(obj.get("content", ""), end="", flush=True)
                        elif msg_type == "final_response":
                            print("\n[final]", obj.get("content", ""))
                            break  # 结束流循环，避免长时间阻塞
                        elif msg_type == "error":
                            print(f"\n[error] {obj.get('message')}")
                            break
                    except Exception as parse_err:
                        print(f"\n[warn] 解析流式数据失败: {parse_err}")
                        break
        else:
            print(f"错误: {response.status_code}")
            print(response.text)
        return True
    except Exception as e:
        logger.error(f"测试对话接口失败: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    user_id = "test_user_001"
    session_id = "test_session_001"

    # 测试简单聊天功能
    content_blocks = {"type":"text", "content":"你好，我是小明，请问你是谁？"}
    print("开始测试简单聊天功能...")
    if test_chat_interface(content_blocks, user_id, session_id):    
        print("\n*===========================================================================================*")
        logger.info("测试简单聊天功能成功")
        print("*===========================================================================================*")
        print("\n\n\n")
    else:
        logger.error(f"测试简单聊天功能失败")

    
    # 测试记忆功能
    content_blocks = {"type":"text", "content":"你还记得我是谁吗？"}
    print("开始测试记忆功能...")
    if test_chat_interface(content_blocks, user_id, session_id):
        print("\n*===========================================================================================*")
        logger.info("测试记忆功能成功")
        print("*===========================================================================================*")
        print("\n\n\n")
    else:
        logger.error(f"测试记忆功能失败")

    # 测试rag功能
    content_blocks = {"type":"text", "content":"我最近正在看《刻意练习》这本书，我记得第五章中提到，在工作中运用刻意练习原则时，有哪三种阻碍练习的错误思维？请详细列出。"}
    print("开始测试rag功能...")
    if test_chat_interface(content_blocks, user_id, session_id):
        print("\n*===========================================================================================*")
        logger.info("测试rag功能成功")
        print("*===========================================================================================*")
        print("\n\n\n")
    else:
        logger.error(f"测试rag功能失败")

    # 测试多模态功能（图片+文本）
    content_blocks = {"type":"text", "content":"请根据上面的图片，描述一下这张图片的内容。"}
    image_path = "test_image.jpg"
    print("开始测试多模态功能（图片+文本）...")
    if test_chat_interface(content_blocks, user_id, session_id, image_file=image_path):
        print("\n*===========================================================================================*")
        logger.info("测试多模态功能（图片+文本）成功")
        print("*===========================================================================================*")
        print("\n\n\n")
    else:
        logger.error(f"测试多模态功能（图片+文本）失败")

    #测试多模态功能（pdf+文本）
    content_blocks = {"type":"text", "content":"请根据pdf文档中的内容，完成任务。"}
    pdf_path = "test_pdf.pdf"
    print("开始测试多模态功能（PDF+文本）...")
    if test_chat_interface(content_blocks, user_id, session_id, pdf_file=pdf_path): 
        print("\n*===========================================================================================*")
        logger.info("测试多模态功能（PDF+文本）成功")
        print("*===========================================================================================*")
        print("\n\n\n")
    else:
        logger.error(f"测试多模态功能（PDF+文本）失败")
    
        

