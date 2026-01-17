# Omni-RAG-QA-ChatBot
使用 LangChain 1.0 搭建的多模态 RAG 问答机器人，支持文本、图片、音频、PDF 输入，基于本地 PostgreSQL 构建知识库，并具备对话记忆管理。

## 主要功能
- 多模态输入：文本 / 图片 / 音频 / PDF，图片和音频以 base64 形式注入 prompt。
- 本地知识库：PostgreSQL + `documents` / `document_chunks`，支持向量相似度与文本相似度混合检索。
- 对话记忆：基于 PostgreSQL 的 `conversations` / `messages`，按 `user_id` + `session_id` 存取，每会话自动保留最近 5 条对话。

### 多模态输入
用户可同时上传文本、图片、音频、PDF；图片和音频由后端编码为 base64 注入模型。

### RAG 检索
文档切分后存入 `document_chunks` 并向量化；查询时做向量+文本相似度检索，将相关片段拼入提示词。

### 对话记忆
使用 `conversations`/`messages` 表管理历史，`user_id` + `session_id` 标识会话；每轮对话自动读取最近 5 条，并在响应后写回最新记录。

## 使用方法
1. 根据 `requirements.txt` 安装依赖。
2. 运行 `setup_db.py` 初始化数据库（创建表和索引）。
3. 运行 `rag_manager.py` 构建本地知识库（向量化文档）。
4. 启动服务：`python main.py`。
5. 测试：`test.py` 提供调用示例，可传文本/图片/音频/PDF。

附注：`main.py` 内的 `chat_sync` 便于本地非接口调试；流式接口为 `/api/chat/stream`。
