"""
数据库初始化脚本
创建所有必要的表结构，包括pgvector扩展
"""

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

# PostgreSQL连接信息
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag")

# 数据库连接URL
DATABASE_URL = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# 建立连接
engine = create_engine(DATABASE_URL)

def init_db():
    """初始化数据库：创建表、扩展等"""
    
    with engine.connect() as conn:
        # 1. 创建pgvector扩展
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            print("✓ pgvector扩展已创建")
        except Exception as e:
            print(f"⚠ pgvector扩展可能已存在: {e}")

        # 1.1 创建pg_trgm扩展（用于全文模糊匹配与相似度）
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            print("✓ pg_trgm扩展已创建")
        except Exception as e:
            print(f"⚠ pg_trgm扩展可能已存在: {e}")
        
        # 2. 创建documents表（企业文档）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                category VARCHAR(100),
                uploaded_by VARCHAR(255),
                file_path VARCHAR(500),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ documents表已创建")
        
        # 3. 创建document_chunks表（文档分块 + 向量）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector(1536),
                page_number INTEGER,
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ document_chunks表已创建")
        
        # 4. 创建向量索引（加速查询）
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            print("✓ 向量索引已创建")
        except Exception as e:
            print(f"⚠ 向量索引创建提示: {e}")
        
        # 5. 创建conversations表（用户会话）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) UNIQUE NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                title VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ conversations表已创建")
        
        # 6. 创建messages表（对话消息）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ messages表已创建")
        
        # 7. 创建retrieval_records表（检索记录 - 用于引用追踪）
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS retrieval_records (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                query TEXT,
                retrieved_chunk_ids INTEGER[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        print("✓ retrieval_records表已创建")
        
        # 8. 创建索引
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_created_at 
            ON documents(created_at)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
            ON document_chunks(document_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id 
            ON conversations(user_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session_id 
            ON conversations(session_id)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id 
            ON messages(conversation_id)
        """))
        # 为全文检索创建 FTS GIN 索引（simple 配置）
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_fts
            ON document_chunks USING gin (to_tsvector('simple', content))
        """))
        # 为文本相似度创建 trigram GIN 索引（pg_trgm）
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_content_trgm
            ON document_chunks USING gin (content gin_trgm_ops)
        """))
        print("✓ 所有索引已创建")
        
        # 提交所有更改
        conn.commit()

if __name__ == "__main__":
    print(f"正在连接数据库: {DATABASE_URL}")
    try:
        init_db()
        print("\n✅ 数据库初始化完成！")
    except Exception as e:
        print(f"\n❌ 数据库初始化失败: {e}")
        raise
