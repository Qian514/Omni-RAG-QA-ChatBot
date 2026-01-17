"""
RAG核心模块
负责向量化、文档分割、检索等RAG相关操作
"""

import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
import logging
from dotenv import load_dotenv
from db_manager import db_manager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """向量化管理器"""
    
    def __init__(self):
        """初始化embedding模型"""
        from langchain_community.embeddings import DashScopeEmbeddings
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    
    def embed_text(self, text: str) -> List[float]:
        """
        将文本转换为向量
        
        Args:
            text: 输入文本
        
        Returns:
            向量（列表形式）
        """
        embedding = self.embeddings.embed_query(text)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化文本
        """
        embeddings = self.embeddings.embed_documents(texts)
        return embeddings


class DocumentProcessor:
    """文档处理和分割"""
    
    def __init__(self):
        """初始化文本分割器"""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,  # 每个chunk大约800字
            chunk_overlap=50,  # 相邻chunks有200字重叠
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
    
    def split_text(self, text: str) -> List[str]:
        """
        分割文本成chunks
        
        Args:
            text: 要分割的文本
        
        Returns:
            chunks列表
        """
        chunks = self.splitter.split_text(text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]


class RAGManager:
    """RAG核心管理器"""
    
    def __init__(self):
        """初始化RAG组件"""
        self.embedding_manager = EmbeddingManager()
        self.document_processor = DocumentProcessor()
        self.chat_model = self._init_chat_model()
    
    def _init_chat_model(self):
        """初始化LLM模型"""
        try:
            model = init_chat_model(
                model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                model_provider="openai",
                base_url="https://api.siliconflow.cn/v1/",
                api_key=os.getenv("GUIJI_API_KEY"),
            )
            logger.info("✓ LLM模型初始化成功")
            return model
        except Exception as e:
            logger.error(f"❌ LLM模型初始化失败: {e}")
            raise
    
    # ==================== 第一步：知识库构建 ====================
    
    def ingest_document(self, document_title: str, document_content: str,
                       category: str = "general", uploaded_by: str = "system",
                       file_path: str = None) -> int:
        """
        摄取文档到知识库
        
        流程：
        1. 将文档存入documents表
        2. 分割文档成chunks
        3. 向量化每个chunk
        4. 将chunks + 向量存入document_chunks表
        
        Args:
            document_title: 文档标题
            document_content: 文档内容
            category: 文档分类
            uploaded_by: 上传者
            file_path: 文件路径
        
        Returns:
            document_id
        """
        logger.info(f"开始摄取文档: {document_title}")
        
        # 第1步：存入documents表
        doc_id = db_manager.add_document(
            title=document_title,
            content=document_content,
            category=category,
            uploaded_by=uploaded_by,
            file_path=file_path
        )
        
        # 第2步：分割文本
        chunks = self.document_processor.split_text(document_content)
        logger.info(f"文档分割成 {len(chunks)} 个chunks")
        
        # 第3步：批量向量化
        logger.info(f"正在向量化 {len(chunks)} 个chunks...")
        embeddings = self.embedding_manager.embed_texts(chunks)
        
        # 第4步：存入数据库
        for chunk_index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
            db_manager.add_chunk(
                document_id=doc_id,
                chunk_index=chunk_index,
                content=chunk_text,
                embedding=embedding,
                page_number=None
            )
        
        logger.info(f"✓ 文档摄取完成: id={doc_id}, chunks={len(chunks)}")
        return doc_id
    
    # ==================== 第二步：知识库检索 ====================
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        检索与查询最相关的chunks
        
        流程：
        1. 向量化查询
        2. 在PostgreSQL中进行向量相似度搜索
        3. 返回最相关的chunks
        
        Args:
            query: 用户查询文本
            top_k: 返回前K个最相关的chunks
        
        Returns:
            chunks列表（按相似度排序）
        """
        logger.info(f"开始检索查询: {query[:100]}...")
        
        # 第1步：向量化查询
        query_embedding = self.embedding_manager.embed_text(query)
        
        # 第2步：混合检索（向量 + 文本三元组相似度）
        relevant_chunks = db_manager.search_hybrid_chunks(
            query_text=query,
            query_embedding=query_embedding,
            top_k=top_k,
            vec_threshold=0.5,
            weight_vec=0.7,
            weight_text=0.3,
        )
        
        logger.info(f"检索到 {len(relevant_chunks)} 个相关chunks")
        return relevant_chunks
    
    # ==================== 第三步：生成LLM回复 ====================
    
    def generate_rag_response(self, query: str, conversation_history: List[Dict] = None) -> str:
        """
        生成RAG增强的LLM回复
        
        流程：
        1. 检索相关chunks
        2. 构建RAG prompt
        3. 调用LLM生成回复
        
        Args:
            query: 用户查询
            conversation_history: 对话历史（可选）
        
        Returns:
            LLM生成的回复
        """
        # 第1步：检索相关信息
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=5)
        
        # 第2步：构建context
        context = self._build_rag_context(relevant_chunks)
        
        # 第3步：构建messages
        messages = self._build_messages(query, context, conversation_history)
        
        # 第4步：调用LLM
        response = self.chat_model.invoke(messages)
        
        return response.content, relevant_chunks
    
    def _build_rag_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        从检索到的chunks构建context字符串
        """
        if not chunks:
            return "（未找到相关参考文档）"
        
        context = "参考文档内容：\n"
        for idx, chunk in enumerate(chunks, 1):
            doc_title = chunk.get("document_title", "未知")
            content = chunk.get("content", "")
            score = chunk.get("score")
            vec_sim = chunk.get("vector_similarity")
            txt_sim = chunk.get("text_similarity")
            # 兼容旧字段
            similarity = chunk.get("similarity", score if score is not None else 0)
            page = f"(第{chunk.get('page_number')}页)" if chunk.get('page_number') else ""
            
            metrics = []
            if score is not None:
                metrics.append(f"综合分数: {score:.3f}")
            metrics.append(f"相似度: {similarity:.3f}")
            if vec_sim is not None:
                metrics.append(f"向量: {vec_sim:.3f}")
            if txt_sim is not None:
                metrics.append(f"文本: {txt_sim:.3f}")

            context += f"\n[{idx}] 来源: {doc_title} {page}\n" + " | ".join(metrics) + f"\n内容: {content}\n"
        
        return context
    
    def _build_messages(self, query: str, context: str, 
                        conversation_history: List[Dict] = None) -> List:
        """
        构建发送给LLM的messages
        """
        system_prompt = """
你是一个企业知识库助手。你的职责是：
1. 根据提供的参考文档回答用户问题
2. 在回答中明确引用参考文档，使用[1]、[2]等格式
3. 如果参考文档中没有相关信息，明确说明
4. 保持专业、准确、友好的语气

重要：必须在回答中标注引用，格式如：这个信息来自[1]。
"""
        
        messages = [SystemMessage(content=system_prompt)]
        
        # 添加历史消息
        if conversation_history:
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    from langchain.schema import AIMessage
                    messages.append(AIMessage(content=content))
        
        # 添加当前查询（包含RAG context）
        current_message = f"""
参考文档：
{context}

用户问题：{query}

请根据上述参考文档回答用户的问题。
"""
        messages.append(HumanMessage(content=current_message))
        
        return messages


# 创建全局实例
rag_manager = RAGManager()

if __name__ == "__main__":
    # 添加pdf文档
    import fitz
    doc = fitz.open("刻意练习.pdf")
    full_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        full_text += page.get_text()
    try:
        rag_manager.ingest_document(
            document_title="刻意练习",
            document_content=full_text)
        logger.info("文档摄取测试成功")
    except Exception as e:
        logger.error(f"文档摄取测试失败: {e}")

    
