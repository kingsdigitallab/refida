import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from txtai.pipeline import Textractor

from settings import PARAGRAPH_EXCLUDE_PATTERN, UOA, UOA_PATTERN


def extract(
    files: Iterator[Path],
) -> pd.DataFrame:
    extracted = pd.DataFrame()

    for file in files:
        data = extract_file(file)
        extracted = pd.concat([data, extracted], ignore_index=True)

    return extracted


def extract_file(file: Path) -> pd.DataFrame:
    """Extract data from a file using txtai.Textractor.
    :param file: Path to the file to extract data from."""
    textractor = Textractor(paragraphs=True)
    paragraphs = textractor(file.as_posix())

    data = dict(
        name=file.name.replace(".pdf", ""),
        uoa=None,
        title=None,
        research_start=np.NaN,
        research_end=np.NaN,
        impact_start=np.NaN,
        impact_end=np.NaN,
        summary=None,
        details=None,
        sources=None,
        file=file.as_posix(),
    )

    uoa_index = get_paragraph_index(paragraphs, "unit of assessment")
    if uoa_index:
        uoa = get_uoa(paragraphs[uoa_index])
        if uoa:
            data["uoa"] = uoa

    title_index = get_paragraph_index(paragraphs, "title of case study")
    if title_index:
        title = paragraphs[title_index].split(": ")[1]
        data["title"] = title

    for period_title in [
        ("research", "period when the underpinning research was undertaken"),
        ("impact", "period when the claimed impact occurred"),
    ]:
        period_index = get_paragraph_index(paragraphs, period_title[1])
        if period_index:
            text = paragraphs[period_index]
            if text:
                period = get_period(text)
                if period:
                    data[f"{period_title[0]}_start"] = period[0]
                    if len(period) > 1:
                        data[f"{period_title[0]}_end"] = period[1]

    for section in [
        ("summary", "1. summary of the impact", "2. underpinning research"),
        ("details", "4. details of the impact", "5. sources to corroborate the impact"),
        ("sources", "5. sources to corroborate the impact", paragraphs[-1]),
    ]:
        start_index = get_paragraph_index(paragraphs, section[1])
        end_index = get_paragraph_index(paragraphs, section[2])

        if start_index and end_index:
            start_index += 1
            if section[0] == "sources":
                end_index += 1

            content = [
                p for p in paragraphs[start_index:end_index] if include_paragraph(p)
            ]

            data[section[0]] = "\n".join(content)

    return pd.DataFrame([data])


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


def get_uoa(
    text: str, pattern: re.Pattern = UOA_PATTERN, units: dict[str, str] = UOA
) -> Optional[str]:
    """
    Get the Unit of Assessment number from a string.

    :param text: The string to extract the UoA from.
    :param pattern: The regex pattern to use to extract the UoA from the string.
    :param units: A dictionary of Units of assessment ids to their names.
    """
    if not text:
        return None

    match = pattern.search(text)
    if match:
        return units[match.group(1)]

    return None


def get_period(text: str) -> Optional[list[str]]:
    """
    Get the start and end years from a string.

    :param text: The string to extract the years from.
    """
    if not text:
        return None

    return re.findall(r"\d{4}", text)


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
