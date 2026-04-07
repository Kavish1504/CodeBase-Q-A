from __future__ import annotations
import re
from langchain.schema import Document
from loguru import logger
from dataclasses import dataclass,field
from typing import List



_BLOCK_PATTERNS: list[re.Pattern] = [
    # Python
    re.compile(r"^(async\s+)?def\s+\w+", re.MULTILINE),
    re.compile(r"^class\s+\w+", re.MULTILINE),
    # JavaScript / TypeScript
    re.compile(r"^(export\s+)?(default\s+)?(async\s+)?function\s+\w+", re.MULTILINE),
    re.compile(r"^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s+)?\(", re.MULTILINE),
    re.compile(r"^(export\s+)?class\s+\w+", re.MULTILINE),
    # Java / Kotlin / C# — method-like
    re.compile(r"^\s{0,4}(public|private|protected|static|override)\s+\S+\s+\w+\s*\(", re.MULTILINE),
    # Go
    re.compile(r"^func\s+", re.MULTILINE),
    # Rust
    re.compile(r"^(pub\s+)?(async\s+)?fn\s+\w+", re.MULTILINE),
]

_MAX_CHUNK_LINES = 80 
_MIN_CHUNK_LINES = 5 

@dataclass
class _Block:
    start_line=0
    lines:list[str]=field(default_factory=list)

    @property
    def end_line(self)->int:
        return self.start_line+len(self.lines)-1
    
    def text(self):
        return "\n".join(self.lines)
    
def _find_block_boundaries(content:str)->list[int]:
    boundaries:set[int]={0}
    for pattern in _BLOCK_PATTERNS:
        for match in pattern.finditer(content):
            index=content[:match.start()].count("\n")
            boundaries.add(index)
    return sorted(boundaries)

def _make_doc(lines:list[str],start_line:int,file_path:str,language:str)->Document|None:
    text="\n".join(lines).strip()
    if not text:
        return None
    end_line = start_line + len(lines) - 1
    return Document(
        page_content=text,
        metadata={
            "file_path": file_path,
            "language": language,
            "start_line": start_line,
            "end_line": end_line,
            "source": f"{file_path}:{start_line}-{end_line}",
        },
    )



def chunk_file(file_dict:dict)->list[Document]:
    content=file_dict["content"]
    language=file_dict["language"]
    file_path=file_dict["file_path"]
    documents:list[Document]=[]
    if not content or not content.strip():
        return []

    lines=content.splitlines()
    boundaries=_find_block_boundaries(content)
    for i,boundary in enumerate(boundaries):
        next_boundary=boundaries[i+1] if i+1 <len(boundaries) else len(lines)
        block_lines=lines[boundary:next_boundary]
        if len(block_lines)<_MIN_CHUNK_LINES:
            continue
        
        if len(block_lines)>_MAX_CHUNK_LINES:
            step=_MAX_CHUNK_LINES-10
            for offset in range(0,len(block_lines),step):
                sub_lines=block_lines[offset:offset+_MAX_CHUNK_LINES]
                if len(sub_lines)<_MIN_CHUNK_LINES:
                    break
                doc = _make_doc(sub_lines, boundary + offset + 1, file_path, language)
                if doc:
                    documents.append(doc)
        else:
            doc = _make_doc(block_lines, boundary + 1, file_path, language)
            if doc:
                documents.append(doc)

    if not documents:
            logger.debug(f"No blocks found in {file_path}, using line-window fallback.")
            step=_MAX_CHUNK_LINES-10
            for offset in range(0, len(lines), step):
                chunk_lines = lines[offset: offset + _MAX_CHUNK_LINES]
                doc = _make_doc(chunk_lines, offset + 1, file_path, language)
                if doc:
                    documents.append(doc)

    return documents


    
def chunk_all_files(files: list[dict]) -> list[Document]:
    """Chunk every file and return a flat list of Documents."""
    all_docs: list[Document] = []
    for f in files:
        try:
            docs = chunk_file(f)
            all_docs.extend(docs)
        except Exception as exc:
            logger.warning(f"Failed to chunk {f['file_path']}: {exc}")
    logger.info(f"Total chunks produced: {len(all_docs)}")
    return all_docs


        



    
