import fitz
import base64
import re
from pathlib import Path
from typing import Optional
from app.core.logger import logger
from app.core.exceptions import PDFParseError
from app.models.schemas import Paper


class PDFParser:

    def __init__(self):
        self._references_start_line = 0
        logger.info("PDFParser initialized")

    def parse(self, paper: Paper) -> dict:
        logger.info(f"Parsing PDF: {paper.paper_id}")
        if not paper.pdf_path:
            raise PDFParseError(f"No PDF path for paper {paper.paper_id}")
        if not Path(paper.pdf_path).exists():
            raise PDFParseError(f"PDF not found at {paper.pdf_path}")
        try:
            doc = fitz.open(paper.pdf_path)
            result = {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "authors": paper.authors,
                "abstract": paper.abstract,
                "published": paper.published,
                "pages": [],
                "images": [],
                "tables": [],
                "references": None
            }
            self._references_start_line = 0
            in_references = False

            for page_num, page in enumerate(doc, start=1):
                text = page.get_text().strip()
                lines = text.split("\n")

                if not in_references and self._is_references_section(text, page_num):
                    in_references = True

                    content_part = "\n".join(
                        lines[:self._references_start_line]
                    ).strip()
                    references_part = "\n".join(
                        lines[self._references_start_line:]
                    ).strip()

                    if content_part:
                        tables = self._extract_tables(page, page_num)
                        result["tables"].extend(tables)
                        clean_text = self._remove_table_text(tables, content_part)
                        if clean_text:
                            result["pages"].append({
                                "page_number": page_num,
                                "text": clean_text
                            })

                    result["references"] = references_part
                    continue

                if in_references:
                    result["references"] += "\n" + text
                    continue

                tables = self._extract_tables(page, page_num)
                result["tables"].extend(tables)
                clean_text = self._remove_table_text(tables, text)
                if clean_text:
                    result["pages"].append({
                        "page_number": page_num,
                        "text": clean_text
                    })
                images = self._extract_images(doc, page, page_num, text)
                result["images"].extend(images)

            doc.close()
            logger.info(
                f"Parsed {paper.paper_id}: "
                f"{len(result['pages'])} pages, "
                f"{len(result['images'])} images, "
                f"{len(result['tables'])} tables, "
                f"references: {'yes' if result['references'] else 'no'}"
            )
            return result

        except PDFParseError:
            raise
        except Exception as e:
            raise PDFParseError(f"Failed to parse {paper.paper_id}: {e}")

    def _is_references_section(self, text: str, page_num: int) -> bool:
        if page_num <= 3:
            return False
        lines = text.strip().split("\n")
        reference_headers = [
            "references",
            "bibliography",
            "works cited",
            "reference",
        ]
        for i, line in enumerate(lines):
            clean = " ".join(line.lower().strip().split())
            if len(clean) > 20:
                continue
            for header in reference_headers:
                if clean == header:
                    self._references_start_line = i
                    return True
                if clean.endswith(header) and len(clean) < 15:
                    self._references_start_line = i
                    return True
        return False

    def _extract_tables(self, page, page_num: int) -> list[dict]:
        tables = []
        try:
            table_finder = page.find_tables()
            for table in table_finder.tables:
                rows = table.extract()
                if rows:
                    table_text = "\n".join(
                        " | ".join(str(cell) for cell in row if cell)
                        for row in rows
                    )
                    caption = self._find_caption(
                        page.get_text(),
                        "table",
                        len(tables) + 1
                    )
                    tables.append({
                        "page_number": page_num,
                        "content": table_text,
                        "caption": caption
                    })
        except Exception:
            pass
        return tables

    def _extract_images(self, doc, page, page_num: int, page_text: str) -> list[dict]:
        images = []
        try:
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) < 5000:
                    continue
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                caption = self._find_caption(
                    page_text,
                    "figure",
                    len(images) + 1
                )
                images.append({
                    "page_number": page_num,
                    "image_b64": image_b64,
                    "ext": base_image["ext"],
                    "caption": caption,
                    "description": None
                })
        except Exception as e:
            logger.warning(f"Image extraction issue on page {page_num}: {e}")
        return images

    def _find_caption(self, text: str, caption_type: str, number: int) -> Optional[str]:
        patterns = [
            rf"(?i){caption_type}\s*{number}[.:]\s*(.+?)(?:\n|$)",
            rf"(?i){caption_type}\s*{number}\s+(.+?)(?:\n|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return None

    def _remove_table_text(self, tables: list[dict], text: str) -> str:
        clean = text
        for table in tables:
            clean = clean.replace(table["content"], "").strip()
        return clean