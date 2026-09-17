"""文档处理与知识库构建管线 —— 方案 7.4 的实现。

功能：
  1. 解析 TXT / MD / (PDF/DOCX 桩) 文档
  2. 按章节/标题结构化切分
  3. 元数据标注与冲突检测
  4. 批量嵌入与注册到 RagClient
  5. 预览和校验

使用示例:
    from railmind.core.kb_builder import DocumentProcessor
    from railmind.core.rag import RagClient

    processor = DocumentProcessor()
    chunks = processor.process_file("检修规程.md", asset_type="pantograph")
    client = RagClient()
    client.add_chunks(chunks)
"""

from __future__ import annotations

import os
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from railmind.core.rag import KBChunk, VERSION_DRAFT, VERSION_ACTIVE

# ---------------------------------------------------------------------------
# 配置常量
# ---------------------------------------------------------------------------

# 切分参数
CHUNK_MIN_CHARS = 200       # 最小块字符数（较短的合并到前一块）
CHUNK_TARGET_CHARS = 600    # 目标块字符数
CHUNK_MAX_CHARS = 1000      # 最大块字符数（超出强制分割）
CHUNK_OVERLAP_CHARS = 80    # 相邻块重叠字符数

# 支持的文档类型
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".pdf", ".docx"}


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class ParseError(Exception):
    """文档解析错误。"""


class ValidationError(Exception):
    """元数据校验错误。"""


# ---------------------------------------------------------------------------
# 文档元数据
# ---------------------------------------------------------------------------


@dataclass
class DocumentMeta:
    """待处理文档的元数据（从文件名/用户输入推断）。"""
    title: str
    source: str = ""             # 来源（如 "用户上传"）
    document_type: str = ""      # "规程"/"说明书"/"案例"
    version: str = "1.0"
    asset_type: str = ""         # 适用资产类型
    fault_types: List[str] = field(default_factory=list)
    train_models: List[str] = field(default_factory=list)
    permission: str = "INTERNAL"
    author: str = ""


# ---------------------------------------------------------------------------
# 文本解析器
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """清理文本：统一换行、去除多余空白。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)  # 多个空行压缩为两个
    text = text.strip()
    return text


def _detect_headings(text: str) -> List[Tuple[int, str, str]]:
    """检测 Markdown 风格的标题行。

    Returns:
        List[(level, title_text, full_line)]，level 1-6。
    """
    headings: List[Tuple[int, str, str]] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append((level, title, line))
        # 也检测中文序号标题（如 "一、", "1.1", "第X章"）
        elif re.match(r"^第[一二三四五六七八九十百]+[章节篇条]\s+", stripped):
            headings.append((2, stripped, line))
        elif re.match(r"^[一二三四五六七八九十]+[、.．]\s*", stripped):
            headings.append((3, stripped, line))
        elif re.match(r"^\d+\.\d+\s", stripped):
            headings.append((4, stripped, line))
    return headings


def _parse_markdown(text: str) -> List[Dict[str, Any]]:
    """解析 Markdown 文本为结构化的章节段落。

    Returns:
        List[{"heading": str, "level": int, "content": str, "page": int}]。
    """
    text = _clean_text(text)
    headings = _detect_headings(text)

    if not headings:
        # 无标题结构，整篇作为一个章节
        return [{"heading": "", "level": 0, "content": text, "page": 1}]

    sections: List[Dict[str, Any]] = []
    lines = text.split("\n")
    current_heading = ""
    current_level = 0
    current_content: List[str] = []
    # 标题行索引
    heading_line_indices = set()
    for _, _, hl in headings:
        for i, line in enumerate(lines):
            if line == hl:
                heading_line_indices.add(i)
                break

    for i, line in enumerate(lines):
        stripped = line.strip()
        is_heading = any(
            stripped == hl for _, _, hl in headings
        )

        if is_heading and current_content:
            # 保存上一节
            content = "\n".join(current_content).strip()
            if content:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "content": content,
                    "page": 1,
                })
            current_content = []
            # 提取新标题
            for level, title, hl in headings:
                if stripped == hl:
                    current_heading = title
                    current_level = level
                    break
        elif is_heading:
            for level, title, hl in headings:
                if stripped == hl:
                    current_heading = title
                    current_level = level
                    break
        else:
            current_content.append(line)

    # 最后一节
    content = "\n".join(current_content).strip()
    if content:
        sections.append({
            "heading": current_heading,
            "level": current_level,
            "content": content,
            "page": 1,
        })

    return sections


def _parse_plain_text(text: str) -> List[Dict[str, Any]]:
    """解析纯文本：按空行分段。"""
    text = _clean_text(text)
    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return []
    # 合并短段落
    merged: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) < CHUNK_TARGET_CHARS:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return [{"heading": "", "level": 0, "content": p, "page": 1} for p in merged]


def _parse_html(text: str) -> List[Dict[str, Any]]:
    """简易 HTML 解析：提取 <h1-6> 和 <p> 文本。"""
    text = _clean_text(text)
    sections: List[Dict[str, Any]] = []
    current_heading = ""
    current_level = 0
    # 提取标题
    for m in re.finditer(r"<h([1-6])[^>]*>(.*?)</h\1>", text, re.IGNORECASE | re.DOTALL):
        level = int(m.group(1))
        title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
        sections.append({"heading": title, "level": level, "content": "", "page": 1})
        current_heading = title
        current_level = level
    # 提取段落
    paras = re.findall(r"<p[^>]*>(.*?)</p>", text, re.IGNORECASE | re.DOTALL)
    content = "\n\n".join(re.sub(r"<[^>]+>", "", p).strip() for p in paras if p.strip())
    if content and not sections:
        sections.append({"heading": "", "level": 0, "content": content, "page": 1})
    elif content and sections:
        sections[-1]["content"] = content
    return sections


def parse_text(text: str, source_format: str = "md") -> List[Dict[str, Any]]:
    """按格式解析文本为结构化章节。"""
    if source_format == "md":
        return _parse_markdown(text)
    elif source_format == "html":
        return _parse_html(text)
    elif source_format == "txt":
        return _parse_plain_text(text)
    else:
        return _parse_plain_text(text)


# ---------------------------------------------------------------------------
# 分块器
# ---------------------------------------------------------------------------


def _split_long_content(content: str, heading: str) -> List[str]:
    """将过长的内容按目标字符数切分，带重叠。"""
    if len(content) <= CHUNK_MAX_CHARS:
        return [content]

    chunks: List[str] = []
    start = 0
    while start < len(content):
        end = min(start + CHUNK_TARGET_CHARS, len(content))
        if end < len(content):
            # 尽量在句号/换行处断开
            cut = content.rfind("。", start, end)
            if cut > start + CHUNK_MIN_CHARS:
                end = cut + 1
            else:
                cut = content.rfind("\n", start, end)
                if cut > start + CHUNK_MIN_CHARS:
                    end = cut + 1
        if end <= start:
            # 安全防护：无法推进时强制前移
            end = min(start + CHUNK_TARGET_CHARS, len(content))
            if end <= start:
                break
        chunk_text = content[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        next_start = end - CHUNK_OVERLAP_CHARS
        if next_start <= start or next_start >= len(content):
            break
        start = next_start
    return chunks


def chunk_sections(
    sections: List[Dict[str, Any]],
    document_id: str,
    title: str,
    asset_type: str,
    fault_types: List[str],
    version: str = "1.0",
    permission: str = "INTERNAL",
    tags: Optional[List[str]] = None,
) -> List[KBChunk]:
    """将结构化章节列表切分为 KBChunk 列表。"""
    now = time.time()
    chunks: List[KBChunk] = []
    page_counter = 1

    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "")
        page = sec.get("page", page_counter)

        if not content.strip():
            continue

        # 长内容分割
        parts = _split_long_content(content, heading)
        for part in parts:
            text = part.strip()
            if not text:
                continue
            # 标题放在前面
            full_text = f"{heading}：{text}" if heading else text
            chunk = KBChunk(
                document_id=document_id,
                title=title,
                chapter=heading or "正文",
                page=page,
                text=full_text,
                asset_type=asset_type,
                fault_type=list(fault_types),
                version=version,
                effective_status=VERSION_ACTIVE,
                permission=permission,
                tags=tags or [],
                created_at=now,
                updated_at=now,
            )
            chunks.append(chunk)

        page_counter += 1

    return chunks


# ---------------------------------------------------------------------------
# DocumentProcessor —— 管线入口
# ---------------------------------------------------------------------------


class DocumentProcessor:
    """文档处理管线：解析 → 切分 → 元数据标注 → KBChunk 列表。"""

    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, default_permission: str = "INTERNAL"):
        self.default_permission = default_permission
        self.stats: Dict[str, Any] = {
            "processed": 0,
            "chunks_created": 0,
            "errors": 0,
        }

    def process_text(
        self,
        text: str,
        document_id: str,
        title: str,
        asset_type: str,
        fault_types: Optional[List[str]] = None,
        version: str = "1.0",
        source_format: str = "md",
        permission: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KBChunk]:
        """处理文本内容，返回 KBChunk 列表。"""
        sections = parse_text(text, source_format)
        chunks = chunk_sections(
            sections=sections,
            document_id=document_id,
            title=title,
            asset_type=asset_type,
            fault_types=fault_types or [],
            version=version,
            permission=permission or self.default_permission,
            tags=tags,
        )
        self.stats["processed"] += 1
        self.stats["chunks_created"] += len(chunks)
        return chunks

    def process_file(
        self,
        file_path: str,
        asset_type: str,
        fault_types: Optional[List[str]] = None,
        version: str = "1.0",
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        permission: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[KBChunk]:
        """处理文档文件，返回 KBChunk 列表。

        Args:
            file_path: 文件路径。
            asset_type: 适用资产类型。
            fault_types: 相关故障类型列表。
            version: 版本号。
            document_id: 文档 ID，None 则自动生成。
            title: 文档标题，None 则从文件名推断。
            permission: 权限等级。
            tags: 标签列表。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ParseError(f"不支持的文件类型: {ext}，支持: {SUPPORTED_EXTENSIONS}")

        # 读取文件
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            # 尝试 GBK
            try:
                with open(file_path, "r", encoding="gbk") as f:
                    raw = f.read()
            except Exception as e:
                raise ParseError(f"无法解码文件: {e}")

        # 推断元数据
        doc_id = document_id or self._generate_id(file_path)
        doc_title = title or self._infer_title(file_path)
        fmt = self._format_from_ext(ext)
        perm = permission or self.default_permission

        try:
            chunks = self.process_text(
                text=raw,
                document_id=doc_id,
                title=doc_title,
                asset_type=asset_type,
                fault_types=fault_types,
                version=version,
                source_format=fmt,
                permission=perm,
                tags=tags,
            )
        except Exception as e:
            self.stats["errors"] += 1
            raise ParseError(f"解析失败: {e}")

        return chunks

    # ---- 辅助 ----

    @staticmethod
    def _generate_id(file_path: str) -> str:
        """从文件路径生成稳定的文档 ID。"""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        h = hashlib.md5(file_path.encode()).hexdigest()[:8].upper()
        clean = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff_-]", "_", basename)[:40]
        return f"DOC-{clean}-{h}"

    @staticmethod
    def _infer_title(file_path: str) -> str:
        """从文件名推断标题。"""
        basename = os.path.splitext(os.path.basename(file_path))[0]
        # 移除常见后缀
        title = re.sub(r"[_\-]\d{4}[-_]\d{2}[-_]\d{2}$", "", basename)
        title = re.sub(r"(_v|_版)[.\d]+$", "", title)
        return title.replace("_", "").replace("-", "")

    @staticmethod
    def _format_from_ext(ext: str) -> str:
        mapping = {".md": "md", ".txt": "txt", ".html": "html", ".pdf": "md", ".docx": "md"}
        return mapping.get(ext, "txt")

    # ---- 校验 ----

    @staticmethod
    def validate_chunks(chunks: List[KBChunk]) -> List[str]:
        """校验知识块列表，返回警告列表。"""
        warnings: List[str] = []
        seen_ids = set()
        for i, c in enumerate(chunks):
            if c.document_id in seen_ids:
                warnings.append(f"第 {i} 块 document_id 重复: {c.document_id}")
            seen_ids.add(c.document_id)
            if not c.text or len(c.text) < 10:
                warnings.append(f"第 {i} 块文本过短: {c.document_id}")
            if len(c.text) > 5000:
                warnings.append(f"第 {i} 块文本过长({len(c.text)}字): {c.document_id}")
            if not c.asset_type:
                warnings.append(f"第 {i} 块缺少 asset_type: {c.document_id}")
        return warnings


# ---------------------------------------------------------------------------
# 批量导入工具
# ---------------------------------------------------------------------------


def import_directory(
    directory: str,
    client: Any,  # RagClient
    asset_type_map: Optional[Dict[str, str]] = None,
    default_asset_type: str = "composite_structure",
    recursive: bool = True,
) -> Dict[str, Any]:
    """批量导入目录中的文档到知识库。

    Args:
        directory: 文档目录。
        client: RagClient 实例。
        asset_type_map: 文件名模式到 asset_type 的映射。
        default_asset_type: 默认资产类型。
        recursive: 是否递归扫描子目录。

    Returns:
        处理统计。
    """
    processor = DocumentProcessor()
    total_chunks = 0
    total_files = 0
    errors: List[str] = []

    if recursive:
        iter_scan = [(root, files) for root, _dirs, files in os.walk(directory)]
    else:
        iter_scan = [(directory, os.listdir(directory))]

    for root, files in iter_scan:
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            # 确定 asset_type
            asset_type = default_asset_type
            if asset_type_map:
                for pattern, at in asset_type_map.items():
                    if pattern in fname:
                        asset_type = at
                        break
            try:
                chunks = processor.process_file(fpath, asset_type=asset_type)
                if chunks:
                    client.add_chunks(chunks)
                    total_chunks += len(chunks)
                    total_files += 1
            except Exception as e:
                errors.append(f"{fname}: {e}")

    return {
        "files_processed": total_files,
        "chunks_added": total_chunks,
        "errors": errors,
    }