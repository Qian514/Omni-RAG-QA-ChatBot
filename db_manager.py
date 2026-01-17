"""
数据库管理模块
封装所有数据库操作，包括表的CRUD和向量检索
"""

import os
from typing import List, Dict, Any, Tuple
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import logging

load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PostgreSQL连接信息
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag")

DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self):
        """初始化数据库连接"""
        self.engine = create_engine(DATABASE_URL)
    
    # ==================== Documents 操作 ====================
    
    def add_document(self, title: str, content: str, category: str = None, 
                     uploaded_by: str = None, file_path: str = None) -> int:
        """
        添加文档
        
        Args:
            title: 文档标题
            content: 文档内容
            category: 文档分类
            uploaded_by: 上传者
            file_path: 文件路径
        
        Returns:
            document_id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO documents (title, content, category, uploaded_by, file_path)
                VALUES (:title, :content, :category, :uploaded_by, :file_path)
                RETURNING id
            """), {
                "title": title,
                "content": content,
                "category": category,
                "uploaded_by": uploaded_by,
                "file_path": file_path
            })
            doc_id = result.scalar()
            conn.commit()
            logger.info(f"Document added: id={doc_id}, title={title}")
            return doc_id
    
    def get_document(self, document_id: int) -> Dict[str, Any]:
        """获取文档"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, title, content, category, uploaded_by, file_path, created_at
                FROM documents
                WHERE id = :id
            """), {"id": document_id})
            row = result.fetchone()
            if row:
                return {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "category": row[3],
                    "uploaded_by": row[4],
                    "file_path": row[5],
                    "created_at": row[6]
                }
            return None
    
    def delete_document(self, document_id: int) -> bool:
        """删除文档"""
        with self.engine.connect() as conn:
            conn.execute(text("""
                DELETE FROM documents WHERE id = :id
            """), {"id": document_id})
            conn.commit()
            logger.info(f"Document deleted: id={document_id}")
            return True
    
    # ==================== Document Chunks 操作 ====================
    
    def add_chunk(self, document_id: int, chunk_index: int, content: str, 
                  embedding: List[float], page_number: int = None) -> int:
        """
        添加文档分块（带向量）
        
        Args:
            document_id: 所属文档ID
            chunk_index: 块的索引
            content: 块的文本内容
            embedding: 向量（1536维列表）
            page_number: 页码
        
        Returns:
            chunk_id
        """
        # 将embedding列表转换为PostgreSQL vector格式
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
        
        with self.engine.connect() as conn:
            # CAST(:embedding AS vector) 确保占位符被SQLAlchemy正确替换后再做类型转换
            result = conn.execute(text("""
                INSERT INTO document_chunks 
                (document_id, chunk_index, content, embedding, page_number, token_count)
                VALUES (:doc_id, :chunk_idx, :content, CAST(:embedding AS vector), :page_num, :token_count)
                RETURNING id
            """), {
                "doc_id": document_id,
                "chunk_idx": chunk_index,
                "content": content,
                "embedding": embedding_str,
                "page_num": page_number,
                "token_count": len(content.split())
            })
            chunk_id = result.scalar()
            conn.commit()
            return chunk_id
    
    def get_chunk(self, chunk_id: int) -> Dict[str, Any]:
        """获取单个chunk"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, document_id, chunk_index, content, page_number, created_at
                FROM document_chunks
                WHERE id = :id
            """), {"id": chunk_id})
            row = result.fetchone()
            if row:
                return {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "content": row[3],
                    "page_number": row[4],
                    "created_at": row[5]
                }
            return None
    
    def get_chunks_by_document(self, document_id: int) -> List[Dict[str, Any]]:
        """获取某个文档的所有chunks"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, document_id, chunk_index, content, page_number, created_at
                FROM document_chunks
                WHERE document_id = :doc_id
                ORDER BY chunk_index
            """), {"doc_id": document_id})
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "content": row[3],
                    "page_number": row[4],
                    "created_at": row[5]
                }
                for row in rows
            ]
    
    def delete_document_chunks(self, document_id: int) -> int:
        """删除某个文档的所有chunks"""
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                DELETE FROM document_chunks WHERE document_id = :doc_id
            """), {"doc_id": document_id})
            deleted_count = result.rowcount
            conn.commit()
            logger.info(f"删除 {deleted_count} 个chunks，文档ID={document_id}")
            return deleted_count
    
    # ==================== 向量相似度检索 ====================
    
    def search_similar_chunks(self, query_embedding: List[float], 
                              top_k: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        向量相似度检索
        
        Args:
            query_embedding: 查询向量（1536维）
            top_k: 返回前K个最相似的
            threshold: 相似度阈值（-1到1，值越大越相似）
        
        Returns:
            最相似的chunks列表（包含相似度分数）
        """
        # 将embedding列表转换为PostgreSQL vector格式
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.page_number,
                    d.title,
                    (1 - (dc.embedding <=> CAST(:embedding AS vector))) as similarity
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE (1 - (dc.embedding <=> CAST(:embedding AS vector))) > :threshold
                ORDER BY similarity DESC
                LIMIT :top_k
            """), {
                "embedding": embedding_str,
                "threshold": threshold,
                "top_k": top_k
            })
            
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "content": row[3],
                    "page_number": row[4],
                    "document_title": row[5],
                    "similarity": float(row[6])
                }
                for row in rows
            ]

    def search_hybrid_chunks(self, query_text: str, query_embedding: List[float],
                             top_k: int = 5, vec_threshold: float = 0.1,
                             weight_vec: float = 0.7, weight_text: float = 0.3) -> List[Dict[str, Any]]:
        """
        混合检索：向量相似度 + 文本相似度（pg_trgm）加权融合

        Args:
            query_text: 原始查询文本
            query_embedding: 查询向量
            top_k: 返回数量
            vec_threshold: 向量相似度下限（过滤项）
            weight_vec: 向量相似度权重
            weight_text: 文本相似度权重（pg_trgm similarity）

        Returns:
            按综合得分排序的chunks
        """
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.page_number,
                    d.title,
                    (1 - (dc.embedding <=> CAST(:embedding AS vector))) AS vec_sim,
                    similarity(dc.content, CAST(:q AS text)) AS txt_sim,
                    (:w_vec * (1 - (dc.embedding <=> CAST(:embedding AS vector)))
                     + :w_txt * similarity(dc.content, CAST(:q AS text))) AS score
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE (:w_vec * (1 - (dc.embedding <=> CAST(:embedding AS vector)))
                     + :w_txt * similarity(dc.content, CAST(:q AS text))) > :vec_threshold
                ORDER BY score DESC
                LIMIT :top_k
            """), {
                "embedding": embedding_str,
                "q": query_text,
                "vec_threshold": vec_threshold,
                "top_k": top_k,
                "w_vec": weight_vec,
                "w_txt": weight_text,
            })

            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "document_id": row[1],
                    "chunk_index": row[2],
                    "content": row[3],
                    "page_number": row[4],
                    "document_title": row[5],
                    "vector_similarity": float(row[6]),
                    "text_similarity": float(row[7]),
                    "score": float(row[8])
                }
                for row in rows
            ]
    
    # ==================== Conversations 操作 ====================
    
    def create_conversation(self, session_id: str, user_id: str, 
                           title: str = None) -> int:
        """
        创建新会话
        
        Args:
            session_id: 会话唯一ID
            user_id: 用户ID
            title: 会话标题（可选）
        
        Returns:
            conversation_id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO conversations (session_id, user_id, title)
                VALUES (:session_id, :user_id, :title)
                RETURNING id
            """), {
                "session_id": session_id,
                "user_id": user_id,
                "title": title
            })
            conv_id = result.scalar()
            conn.commit()
            logger.info(f"创建新对话: id={conv_id}, session_id={session_id}")
            return conv_id
    
    def get_conversation_id(self, session_id: str, user_id: str) -> int:
        """
        通过session_id和user_id获取conversation_id
        包含权限验证
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id FROM conversations
                WHERE session_id = :session_id AND user_id = :user_id
            """), {"session_id": session_id, "user_id": user_id})
            row = result.fetchone()
            if row:
                logger.info(f"找到对话ID: session_id={session_id}, user_id={user_id}, conversation_id={row[0]}")
                return row[0]
            else:
                logger.info(f"未找到对话: session_id={session_id}, user_id={user_id}")
                conv_id = self.create_conversation(session_id=session_id, user_id=user_id)
                logger.info(f"已创建新对话: session_id={session_id}, user_id={user_id}, conversation_id={conv_id}")
                return conv_id
            
    
    # ==================== Messages 操作 ====================
    
    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        """
        添加消息到会话
        
        Args:
            conversation_id: 会话ID
            role: 角色（'user' 或 'assistant'）
            content: 消息内容
        
        Returns:
            message_id
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:conv_id, :role, :content)
                RETURNING id
            """), {
                "conv_id": conversation_id,
                "role": role,
                "content": content
            })
            msg_id = result.scalar()
            conn.commit()
            return msg_id
    
    def add_message_with_limit(self, conversation_id: int, role: str, content: str,
                               limit: int = 5) -> int:
        """
        添加消息并保证同一会话最多保留 limit 条最新记录
        """
        with self.engine.begin() as conn:
            result = conn.execute(text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:conv_id, :role, :content)
                RETURNING id
            """), {
                "conv_id": conversation_id,
                "role": role,
                "content": content
            })
            msg_id = result.scalar()

            # 删除超出限制的旧消息，只保留按时间最新的 limit 条
            conn.execute(text("""
                DELETE FROM messages
                WHERE conversation_id = :conv_id
                  AND id NOT IN (
                    SELECT id FROM messages
                    WHERE conversation_id = :conv_id
                    ORDER BY timestamp DESC, id DESC
                    LIMIT :limit
                  )
            """), {
                "conv_id": conversation_id,
                "limit": limit
            })

            return msg_id

    def get_conversation_history(self, conversation_id: int) -> List[Dict[str, Any]]:
        """
        获取会话的完整历史
        
        Args:
            conversation_id: 会话ID
        
        Returns:
            消息列表
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, role, content, timestamp
                FROM messages
                WHERE conversation_id = :conv_id
                ORDER BY timestamp ASC
            """), {"conv_id": conversation_id})
            
            rows = result.fetchall()
            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3]
                }
                for row in rows
            ]

    def get_recent_messages(self, conversation_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取会话的最近消息，默认最多 limit 条
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT id, role, content, timestamp
                FROM messages
                WHERE conversation_id = :conv_id
                ORDER BY timestamp DESC, id DESC
                LIMIT :limit
            """), {"conv_id": conversation_id, "limit": limit})

            rows = result.fetchall()
            # 返回按时间正序，便于直接拼接上下文
            return [
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "timestamp": row[3]
                }
                for row in reversed(rows)
            ]
    
    # ==================== Retrieval Records 操作 ====================
    
    def add_retrieval_record(self, conversation_id: int, query: str, 
                            retrieved_chunk_ids: List[int]) -> int:
        """
        记录检索操作（用于追踪引用）
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                INSERT INTO retrieval_records (conversation_id, query, retrieved_chunk_ids)
                VALUES (:conv_id, :query, :chunk_ids)
                RETURNING id
            """), {
                "conv_id": conversation_id,
                "query": query,
                "chunk_ids": retrieved_chunk_ids
            })
            record_id = result.scalar()
            conn.commit()
            return record_id

# 创建全局实例
db_manager = DatabaseManager()

if __name__ == "__main__":
    # 测试数据库连接
    logger.info(f"Connecting to database at {DATABASE_URL}")
    try:
        with db_manager.engine.connect() as conn:
            logger.info("Database connection successful!")
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")

   