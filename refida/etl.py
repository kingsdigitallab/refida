import re
import zlib
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from txtai.pipeline import Textractor

from refida.models import REFDocument
from settings import (
    PARAGRAPH_CONTENT_EXCLUDE_PATTERN,
    PARAGRAPH_CONTENT_REMOVE,
    PARAGRAPH_TYPE_EXCLUDE_PATTERN,
    UOA,
    UOA_PATTERN,
    get_sections_environment,
    get_sections_impact_case_study,
)


def extract(
    files: Iterator[Path],
) -> pd.DataFrame:
    extracted = pd.DataFrame()

    for file in files:
        data = extract_file(file)
        if data is not None:
            extracted = pd.concat([data, extracted], ignore_index=True)

    return extracted


def extract_file(file: Path) -> Optional[pd.DataFrame]:
    """
    Extract data from a file using txtai.Textractor.

    :param file: Path to the file to extract data from.
    """
    textractor = Textractor(paragraphs=True)

    paragraphs = textractor(file.as_posix())
    if not paragraphs:
        return None

    if not include_paragraph(paragraphs[0], PARAGRAPH_TYPE_EXCLUDE_PATTERN):
        paragraphs = paragraphs[1:]

    full_text = "\n".join([p for p in paragraphs if include_paragraph(p)])

    doc_type = paragraphs[0]

    doc = REFDocument(
        id=file.name.replace(".pdf", ""), type=doc_type, file=file.as_posix()
    )

    uoa = get_uoa(paragraphs)
    if uoa:
        doc.uoa_n = uoa[0]
        doc.uoa = uoa[1]

    title = get_title(paragraphs)
    if not title:
        title = file.name.replace(".pdf", "")
    doc.title = title

    if is_impact_case_study(doc_type):
        names = get_names(paragraphs)
        if names:
            doc.names = names

    if is_impact_case_study(doc_type):
        for section in [
            ("research", "period when the underpinning research was undertaken"),
            ("impact", "period when the claimed impact occurred"),
        ]:
            period = get_period(paragraphs, section[1])
            if period:
                doc.set_field(f"{section[0]}_start", period[0])
                if len(period) > 1:
                    doc.set_field(f"{section[0]}_end", period[1])

    text = ""

    sections = get_sections_impact_case_study(paragraphs[-1])

    if not is_impact_case_study(doc_type):
        sections = get_sections_environment(paragraphs[-1])

    for section in sections:
        content = get_section(paragraphs, section[1], section[2])
        if content:
            text = f"{text}\n{content}"
            doc.set_field(section[0], content)

    doc.text = text

    doc.compressed = zlib.compress(full_text.encode("utf-8"))

    return pd.DataFrame([doc.dict()])


def is_impact_case_study(doc_type: str) -> bool:
    """
    Check if the document type is an impact case study.

    :param doc_type: The document type to check.
    """
    return doc_type.lower().startswith("impact case study")


def include_paragraph(
    paragraph: str, pattern: re.Pattern = PARAGRAPH_CONTENT_EXCLUDE_PATTERN
) -> bool:
    """
    Check if the paragraph matches the given regex pattern.

    :param paragraph: The paragraph to check.
    :param pattern: The regex pattern to check the paragraph against.
    """
    if not paragraph:
        return False

    return not pattern.match(paragraph)


def get_uoa(
    paragraphs: list[str],
    pattern: re.Pattern = UOA_PATTERN,
    units: dict[str, str] = UOA,
) -> Optional[tuple[int, str]]:
    """
    Get the Unit of Assessment.

    :param paragraphs: The list of paragraphs to search through.
    :param pattern: The regex pattern to use to extract the UoA from the string.
    :param units: A dictionary of Units of assessment ids to their names.
    """
    if not paragraphs:
        return None

    index, pos = get_paragraph_index(paragraphs, "unit of assessment:")
    if index >= 0:
        match = pattern.search(paragraphs[index][pos:])
        if match:
            return int(match.group(1)), units[match.group(1)]

    return None


def get_paragraph_index(paragraphs: list[str], start: str) -> tuple[int, int]:
    """
    Get the index of the paragraph and the position within the paragraph that starts
    with the given string.

    :param paragraphs: The list of paragraphs to search through.
    :param start: The string to search for.
    """
    for idx, paragraph in enumerate(paragraphs):
        pos = paragraph.lower().find(start.lower())
        if pos >= 0:
            return idx, pos

    return -1, -1


def get_title(paragraphs: list[str]) -> Optional[str]:
    """
    Get the title from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    """
    if not paragraphs:
        return None

    index, pos = get_paragraph_index(paragraphs, "title of case study:")
    if index >= 0:
        parts = paragraphs[index][pos:].split(": ")
        if len(parts) > 1:
            return parts[1]

    return None


def get_names(paragraphs: list[str]) -> Optional[list[str]]:
    """
    Get the researchers names from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    """
    if not paragraphs:
        return None

    start_index, _ = get_paragraph_index(paragraphs, "name(s):")
    end_index, _ = get_paragraph_index(paragraphs, "role(s)")
    if start_index >= 0 and end_index >= 0:
        start_index += 1
        return paragraphs[start_index:end_index]

    return None


def get_period(paragraphs: list[str], section: str) -> Optional[list[int]]:
    """
    Get the start and end years from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    :param section: The period to search for.
    """
    if not paragraphs:
        return None

    index, _ = get_paragraph_index(paragraphs, section)
    if index >= 0:
        text = paragraphs[index]
        if text:
            return [int(found) for found in re.findall(r"\d{4}", text)]

    return None


def get_section(paragraphs: list[str], start: str, end: Optional[str]) -> Optional[str]:
    """
    Get the section of text from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    :param start: The string to search for to start the section.
    :param end: The string to search for to end the section. Set to None to search
        until the end of the list.
    """
    if not paragraphs:
        return None

    start_index, start_pos = get_paragraph_index(paragraphs, start)
    end_index, end_pos = (
        get_paragraph_index(paragraphs, end) if end else (len(paragraphs), None)
    )

    if end_pos:
        end_pos -= 1

    if start_index >= 0 and end_index >= 0:
        text = ""

        for idx in range(start_index, end_index + 1):
            paragraph = paragraphs[idx]
            if idx == start_index:
                paragraph = paragraph[start_pos:]
            if idx == end_index:
                paragraph = paragraph[:end_pos]
            if include_paragraph(paragraph):
                text = f"{text}\n{PARAGRAPH_CONTENT_REMOVE.sub('', paragraph)}"

        return text.replace(start, "").strip()

    return None
