import re
import zlib
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from txtai.pipeline import Summary, Textractor

from refida.models import REFDocument
from settings import PARAGRAPH_EXCLUDE_PATTERN, UOA, UOA_PATTERN


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

    full_text = "\n".join([p for p in paragraphs if include_paragraph(p)])

    doc = REFDocument(
        id=file.name.replace(".pdf", ""), type=paragraphs[0], file=file.as_posix()
    )

    uoa = get_uoa(paragraphs)
    if uoa:
        doc.uoa = uoa

    title = get_title(paragraphs)
    if title:
        doc.title = title

    names = get_names(paragraphs)
    if names:
        doc.names = names

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

    for section in [
        ("summary", "1. summary of the impact", "2. underpinning research"),
        ("details", "4. details of the impact", "5. sources to corroborate the impact"),
        ("sources", "5. sources to corroborate the impact", paragraphs[-1]),
    ]:
        content = get_section(paragraphs, section[1], section[2])
        if content:
            text = f"{text}\n{content}"
            doc.set_field(section[0], content)

    doc.text = text
    doc.text_summary = summarise(text)

    doc.compressed = zlib.compress(full_text.encode("utf-8"))

    return pd.DataFrame([doc.dict()])


def include_paragraph(
    paragraph: str, pattern: re.Pattern = PARAGRAPH_EXCLUDE_PATTERN
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
) -> Optional[str]:
    """
    Get the Unit of Assessment.

    :param paragraphs: The list of paragraphs to search through.
    :param pattern: The regex pattern to use to extract the UoA from the string.
    :param units: A dictionary of Units of assessment ids to their names.
    """
    if not paragraphs:
        return None

    index = get_paragraph_index(paragraphs, "unit of assessment")
    if index:
        match = pattern.search(paragraphs[index])
        if match:
            return units[match.group(1)]

    return None


def get_paragraph_index(paragraphs: list[str], start: str) -> int:
    """
    Get the index of the paragraph that starts with the given string.

    :param paragraphs: The list of paragraphs to search through.
    :param start: The string to search for.
    """
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.lower().startswith(start.lower()):
            return idx

    return -1


def get_title(paragraphs: list[str]) -> Optional[str]:
    """
    Get the title from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    """
    if not paragraphs:
        return None

    index = get_paragraph_index(paragraphs, "title of case study")
    if index:
        return paragraphs[index].split(": ")[1]

    return None


def get_names(paragraphs: list[str]) -> Optional[list[str]]:
    """
    Get the researchers names from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    """
    if not paragraphs:
        return None

    start_index = get_paragraph_index(paragraphs, "name(s)")
    end_index = get_paragraph_index(paragraphs, "role(s)")
    if start_index and end_index:
        start_index += 1
        return paragraphs[start_index:end_index]

    return None


def get_period(paragraphs: list[str], section: str) -> Optional[list[str]]:
    """
    Get the start and end years from a list of paragraphs.

    :param paragraphs: The list of paragraphs to search through.
    :param section: The period to search for.
    """
    if not paragraphs:
        return None

    index = get_paragraph_index(paragraphs, section)
    if index:
        text = paragraphs[index]
        if text:
            return re.findall(r"\d{4}", text)

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

    start_index = get_paragraph_index(paragraphs, start)
    end_index = get_paragraph_index(paragraphs, end) if end else len(paragraphs)
    if start_index and end_index:
        start_index += 1

        return "\n".join(
            [p for p in paragraphs[start_index:end_index] if include_paragraph(p)]
        )

    return None


def summarise(text: str, name: Optional[str] = None) -> str:
    """
    Summarise the text.

    :param text: The text to summarise.
    :param name: The name of the summary model to use.
    """

    return get_summary_model(name)(text, workers=4)  # type: ignore


summary_model = None


def get_summary_model(name: Optional[str] = None) -> Summary:
    """
    Get the summarisation model.

    :param name: The name of the model to get.
    """
    global summary_model

    if not summary_model:
        summary_model = Summary(name)

    return summary_model
