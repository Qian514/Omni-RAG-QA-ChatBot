import base64
from fastapi import UploadFile, HTTPException
import io
import os
import fitz
from PIL import Image
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ImageProcessor:
    @staticmethod
    def image_to_base64(image_file: UploadFile) -> str:
        try:
            content = image_file.file.read() # file.read() 是UploadFile的方法
            base64_encoded = base64.b64encode(content).decode('utf-8')
            return base64_encoded
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"图像处理失败: {str(e)}")

    @staticmethod
    def get_image_mime_type(filename : str) -> str:
        extension = filename.split('.')[-1].lower()
        mime_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "bmp": "image/bmp",
            "webp": "image/webp"
        }
        return mime_types.get(extension, "image/jpeg")


class AudioProcessor:
    @staticmethod
    def audio_to_base64(audio_file: UploadFile) -> str:
        try:
            # 错误格式筛选
            if audio_file.content_type:
                if audio_file.content_type not in ["audio/mpeg", "audio/wav", "audio/mp4"]:
                    raise HTTPException(status_code=400, detail="不支持的音频格式")
            if audio_file.filename:
                extension = audio_file.filename.split('.')[-1].lower()
                if extension not in ["mp3", "wav", "mp4"]:
                    raise HTTPException(status_code=400, detail="不支持的音频格式")
              
            content = audio_file.file.read()
            if len(content) > 10 * 1024 * 1024:  # 10MB限制
                raise HTTPException(status_code=400, detail="音频文件过大，超过10MB限制")
            
            base64_encoded = base64.b64encode(content).decode('utf-8')
            return base64_encoded
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"音频处理失败: {str(e)}")
    
    @staticmethod
    def get_audio_mime_type(filename : str) -> str:
        extension = filename.split('.')[-1].lower()
        mime_types = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "mp4": "audio/mp4"
        }
        return mime_types.get(extension, "audio/mpeg")

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    async def process_pdf(self, file_content: bytes, file_name: str):
        try:
            if not os.path.exists("temp"):
                os.makedirs("temp")
            temp_path = os.path.join("temp", file_name)
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            doc = fitz.open(temp_path) # 因为fitz不能直接读取bytes，所以先保存为临时文件
            full_text = ""
            page_contents = {}
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                full_text += page.get_text()
                page_contents[page_num + 1] = page.get_text()   # 页码从1开始
            
            chunks = self.text_splitter.split_text(full_text)
            document_chunks = [] # 构建带有元数据的chunks
            for i,chunk in enumerate(chunks):
                if chunk.strip(): # 忽略空chunk
                    # 查找该chunk所属的页码
                    for page_num, content in page_contents.items():
                        if chunk in content:
                            break
                    doc_chunk = {
                        "id" : f"{file_name}_chunk_{i}",
                        "content": chunk.strip(),
                        "metadata":{
                            "source": file_name,
                            "page_number": page_num,
                            "chunk_index":i,
                            "total_chunks": len(chunks),
                            "chunk_length": len(chunk),
                            "referenced_id": f"[{i+1}]",
                            "source_info" : f"文件: {file_name}, 页码: {page_num}"
                        }
                    }
                    document_chunks.append(doc_chunk)
            # 删除临时文件
            os.remove(temp_path)
            return document_chunks
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF保存失败: {str(e)}")
