"""Tools to get resources for oa"""

import os
import dol

# -------------------------------------------------------------------------------------
# Stores

from dol import cache_this

from oa.util import data_files

dflt_resources_dir = str(data_files.parent.parent / "misc" / "data" / "resources")

# -------------------------------------------------------------------------------------
# SSOT tools

_model_info_mapping = {
    "Input": "price_per_million_tokens",  # TODO: Verify that Input fields are always in per-million-token units
    "Output": "price_per_million_tokens_output",  # TODO: Verify that Output fields are always in per-million-token units
}


def pricing_info_persepective_of_model_info():
    from oa.util import pricing_info
    import pandas as pd

    prices_info_ = pd.DataFrame(pricing_info()).drop_duplicates(subset="Model")

    def if_price_change_to_number(price):
        if isinstance(price, str) and price.startswith("$"):
            return float(price.replace("$", "").replace(",", ""))
        elif isinstance(price, dict):
            return {k: if_price_change_to_number(v) for k, v in price.items()}
        else:
            return price

    def ch_field_names(d):
        return {_model_info_mapping.get(k, k): v for k, v in d.items()}

    model_info = {
        row["Model"]: ch_field_names(if_price_change_to_number(row.dropna().to_dict()))
        for _, row in prices_info_.iterrows()
    }

    return model_info


def compare_pricing_info_to_model_info(verbose=True):
    """Compares"""
    from oa._resources import (
        compare_pricing_info_to_model_info,
        pricing_info_persepective_of_model_info,
    )

    from oa.util import pricing_info, model_information_dict

    new_prices = pricing_info_persepective_of_model_info()

    # keys (i.e. model ids) in common with both in model_information_dict & new_prices
    common_keys = set(model_information_dict.keys()) & set(new_prices.keys())

    # # for all keys in common, compare the prices
    for key in common_keys:
        model_info = model_information_dict[key]
        price_info = new_prices[key]

    from lkj import compare_field_values, inclusive_subdict
    from functools import partial

    prices_are_the_same = compare_field_values(
        model_information_dict,
        new_prices,
        default_comparator=compare_field_values,
        aggregator=lambda d: {k: all(v.values()) for k, v in d.items()},
    )
    models_where_prices_are_different = {
        k for k, v in prices_are_the_same.items() if not v
    }

    differences = dict()

    if any(models_where_prices_are_different):
        t = inclusive_subdict(model_information_dict, models_where_prices_are_different)
        tt = inclusive_subdict(new_prices, models_where_prices_are_different)

        differences = compare_field_values(
            t,
            tt,
            default_comparator=partial(
                compare_field_values, default_comparator=lambda x, y: (x, y)
            ),
            # aggregator=lambda d: {k: all(v.values()) for k, v in d.items()},
        )

        if verbose:
            import pprint

            print("Differences:")
            pprint.pprint(differences)

    return differences


# -------------------------------------------------------------------------------------
# Resources class

from dataclasses import dataclass, field
import json
import pathlib
from functools import partial
from typing import Dict, Any, Optional, Callable, Union, List

import pandas as pd
import dol


dflt_pricing_url = "https://platform.openai.com/docs/pricing"


@dataclass
class Resources:
    """
    Data Access class for OpenAI pricing information.

    This class manages the retrieval, computation, and caching of
    OpenAI pricing data through a chain of computations.

    >>> r = Resources()  # doctest: +ELLIPSIS
    >>> r.resources_dir  # doctest: +ELLIPSIS
    '...misc/data/resources'
    """

    resources_dir: str = dflt_resources_dir
    pricing_url: str = dflt_pricing_url

    schema_description_key: str = "openai_api_pricing_schema_description.txt"
    schema_key: str = "api_pricing_schema.json"
    pricing_html_key: str = "openai_api_pricing.html"
    pricing_info_key: str = "openai_api_pricing_info.json"
    pricing_info_from_ai_key: str = "openai_api_pricing_info_from_ai.json"

    # Dependencies that can be injected
    get_pricing_page_html: Optional[Callable[[], str]] = None
    infer_schema_from_verbal_description: Optional[Callable[[str], Dict[str, Any]]] = (
        None
    )
    prompt_json_function: Optional[
        Callable[[str, Dict[str, Any]], Callable[[str], Dict[str, Any]]]
    ] = None

    def __post_init__(self):
        """Initialize stores and ensure dependencies are available."""
        # Set up storage
        if not os.path.exists(self.resources_dir):
            raise FileNotFoundError(f"Directory does not exist: {self.resources_dir}")

        self.json_store = dol.JsonFiles(self.resources_dir)
        self.text_store = dol.TextFiles(self.resources_dir)

        # Import dependencies if not provided
        if self.get_pricing_page_html is None:
            self.get_pricing_page_html = get_pricing_page_html

        if (
            self.infer_schema_from_verbal_description is None
            or self.prompt_json_function is None
        ):
            import oa

            if self.infer_schema_from_verbal_description is None:
                self.infer_schema_from_verbal_description = (
                    oa.infer_schema_from_verbal_description
                )

            if self.prompt_json_function is None:
                self.prompt_json_function = oa.prompt_json_function

    @cache_this(cache="text_store", key=pricing_html_key)
    def pricing_page_html(self):
        """Retrieve the HTML for the OpenAI pricing page."""
        return get_pricing_page_html(self.pricing_url)

    @cache_this(cache="json_store", key=pricing_info_key)
    def pricing_info(self):
        return extract_pricing_data(self.pricing_page_html)

    @dol.cache_this(cache="text_store", key=lambda self: self.schema_description_key)
    def schema_description(self) -> str:
        """Generate and retrieve the schema description for OpenAI API pricing."""
        return """
        A schema to contain the pricing information for OpenAI APIs.
        The first level field should be the name category of API, (things like "Text Tokens", "Audio Tokens", "Fine Tuning", etc.)
        The value of the first level field should be a dictionary with the following fields:
        - "pricing_table_schema": A schema for the pricing table. That is, the list of the columns
        - "pricing_table": A list of dicts, corresponding to the table of pricing information for the API category. 
            For example, ```[{"Model": ..., "Input": ..., ...,}, {"Model": ..., "Input": ..., ...,}, ...]```
            Note that Sometimes there is more than one model name, so we should also have, in there, a field for "Alternate Model Names"
        - "extras": A dictionary with extra information that might be parsed out of each category. 
        For example, there's often some text that gives a description of the API category, and/or specifies "Price per 1M tokens" etc.
            
        Note that some of these tables have a "Batch API" version too. 
        In this case, there should be extra fields that have the same name as the fields above, but with "- Batch" appended to the name.
        For example, "Text Tokens - Batch", "Fine Tuning - Batch", etc.

        It is important to note: These first level fields are not determined in advance. 
        They are determined by the data that is scraped from the page.
        Therefore, these names should not be in the schema. 
        What should be in the schema is the fact that the data should be a JSON whose first 
        level field describes a category, and whose value specifies information about these category.
        """

    @dol.cache_this(cache="json_store", key=lambda self: self.schema_key)
    def schema(self) -> Dict[str, Any]:
        """Generate or retrieve the schema for OpenAI API pricing."""
        return self.infer_schema_from_verbal_description(self.schema_description)

    @dol.cache_this(cache="json_store", key=lambda self: self.pricing_info_from_ai_key)
    def pricing_info_from_ai(self) -> Dict[str, Any]:
        """Extract pricing information from the HTML using AI."""
        prompt = f"""
        Parse through this html and extract the pricing information for the OpenAI APIs.
        The pricing information should be structured according to the schema described below:

        {self.schema_description}

        Here is the html to parse:

        {{html}}
        """

        get_pricing_info = self.prompt_json_function(prompt, self.schema)
        return get_pricing_info(self.pricing_page_html)

    @property
    def pricing_tables_from_ai(self) -> Dict[str, pd.DataFrame]:
        """Convert pricing tables to pandas DataFrames."""
        tables = self.pricing_info_from_ai.get("OpenAI_API_Pricing_Schema", {})

        # Create a mapping interface that transforms table data to DataFrames
        return dol.add_ipython_key_completions(
            dol.wrap_kvs(
                tables, value_decoder=lambda x: pd.DataFrame(x.get("pricing_table", []))
            )
        )

    def list_pricing_categories(self) -> List[str]:
        """List available pricing categories."""
        return list(self.pricing_info_from_ai.get("OpenAI_API_Pricing_Schema", {}))

    def get_pricing_table(self, category: str) -> pd.DataFrame:
        """Get a specific pricing table as a pandas DataFrame."""
        return self.pricing_tables_from_ai[category]


# -------------------------------------------------------------------------------------
# Extract pricing data from HTML content


import re
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


from typing import Dict, List, Optional, Tuple, Any
from bs4 import BeautifulSoup, Tag
import re


def get_pricing_page_html(url=dflt_pricing_url):
    """
    Get the HTML content of the pricing page.

    Need to use chrome to render the page and get the full content.
    Sometimes need to do it twice, since some catpcha pages are shown sometimes.
    """
    from tabled.html import url_to_html_func

    url_to_html = url_to_html_func(("chrome", dict(wait=10)))
    return url_to_html(url)


def parse_pricing_page(html_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the HTML content and extract pricing information from all sections.

    Args:
        html_content: HTML content as a string

    Returns:
        Dictionary with category names as keys and their parsed data as values
    """
    soup = BeautifulSoup(html_content, "html.parser")
    sections = soup.find_all("section")

    results = {}

    for section in sections:
        parsed_categories = parse_section(section)
        if parsed_categories:
            for category_name, data in parsed_categories:
                results[category_name] = data

    return results


def parse_section(section: Tag) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parse a section containing pricing information into potentially multiple categories.

    Args:
        section: BeautifulSoup Tag containing a section

    Returns:
        List of tuples, each with (category_name, data_dict)
    """
    # Find the heading that contains the section name
    heading = section.find("h3", class_="anchor-heading")
    if not heading:
        return []

    # Extract the section name from the heading
    section_name = _clean_text(heading.get_text(strip=True))
    # Remove the "anchor-heading-icon" content if it exists
    svg = heading.find("svg")
    if svg:
        svg_text = svg.get_text(strip=True)
        section_name = section_name.replace(svg_text, "").strip()

    # Find all potential category labels within this section
    category_divs = section.find_all("div", class_="font-medium")

    # If no explicit categories are found, use the section name as the sole category
    if not category_divs:
        category_names = [section_name]
    else:
        # Fix for Image generation section which has a problematic structure
        if section_name == "Image generation":
            category_names = [section_name]
        else:
            category_names = [
                f"{section_name} - {_clean_text(div.get_text(strip=True))}"
                for div in category_divs
            ]

    # Find tables in this section
    tables = section.find_all("table")

    if not tables:
        return []

    # If we have more tables than categories, add generic category names
    if len(tables) > len(category_names):
        for i in range(len(category_names), len(tables)):
            category_names.append(f"{section_name} - Table {i+1}")

    results = []

    # Process each table with its corresponding category name
    for i, (table, category_name) in enumerate(zip(tables, category_names)):
        # Initialize data dictionary for this category
        data = {"pricing_table_schema": [], "pricing_table": [], "extras": {}}

        # Extract any extra information
        extras = _extract_extras(section)
        if extras:
            data["extras"] = extras

        # Extract table schema (column headers)
        thead = table.find("thead")
        if not thead:
            continue

        header_row = thead.find("tr")
        if not header_row:
            continue

        headers = header_row.find_all("th")
        schema = [_clean_text(header.get_text(strip=True)) for header in headers]

        data["pricing_table_schema"] = schema

        # Extract table rows
        tbody = table.find("tbody")
        if not tbody:
            continue

        rows = tbody.find_all("tr")
        table_data = []

        current_row_data = None
        rowspan_active = False
        rowspan_value = 0

        for row in rows:
            row_data = {}
            cells = row.find_all("td")

            # Check if this row is part of a rowspan
            first_cell = cells[0] if cells else None
            if (
                first_cell
                and first_cell.has_attr("rowspan")
                and int(first_cell["rowspan"]) > 1
            ):
                current_row_data = {}  # Start a new rowspan group
                rowspan_active = True
                rowspan_value = int(first_cell["rowspan"])

                # Extract model information from the rowspan cell
                model_info = _extract_model_info(first_cell)
                for key, value in model_info.items():
                    current_row_data[key] = value

                # Process the rest of the cells in this first rowspan row
                for i, cell in enumerate(cells[1:], 1):
                    if i < len(schema):
                        column_name = schema[i]
                        cell_data = _extract_cell_data(cell)
                        row_data[column_name] = cell_data

                # Combine model info with row data
                combined_data = {**current_row_data, **row_data}
                table_data.append(combined_data)
                rowspan_value -= 1

            elif rowspan_active and current_row_data:
                # This is a continuation row in a rowspan
                for i, cell in enumerate(cells):
                    # Adjusted index because first column is handled by rowspan
                    col_idx = i + 1
                    if col_idx < len(schema):
                        column_name = schema[col_idx]
                        cell_data = _extract_cell_data(cell)
                        row_data[column_name] = cell_data

                # Combine the current row data with the continuing rowspan data
                combined_data = {**current_row_data, **row_data}
                table_data.append(combined_data)

                rowspan_value -= 1
                if rowspan_value <= 0:
                    rowspan_active = False

            else:
                # Normal row without rowspan
                rowspan_active = False
                for i, cell in enumerate(cells):
                    if i < len(schema):
                        column_name = schema[i]

                        if i == 0:  # First column is typically the model name
                            model_info = _extract_model_info(cell)
                            for key, value in model_info.items():
                                row_data[key] = value
                        else:
                            cell_data = _extract_cell_data(cell)
                            row_data[column_name] = cell_data

                table_data.append(row_data)

        data["pricing_table"] = table_data
        results.append((category_name, data))

    return results


def _extract_extras(section: Tag) -> Dict[str, str]:
    """Extract extra information from a section."""
    extras = {}

    # Look for descriptive text
    description = section.find(
        "div", class_="text-xs text-gray-500 whitespace-pre-line"
    )
    if description:
        extras["description"] = _clean_text(description.get_text(strip=True))

    return extras


def _extract_model_info(cell: Tag) -> Dict[str, Any]:
    """Extract model name and any alternate model names from a cell."""
    info = {}

    # Find the main model name
    model_div = cell.find("div", class_="text-gray-900")
    if model_div:
        info["Model"] = _clean_text(model_div.get_text(strip=True))

    # Check for alternate model names (usually in a smaller text below)
    alt_model = cell.find("div", class_="text-xs text-gray-600")
    if alt_model:
        info["Alternate_Model"] = _clean_text(alt_model.get_text(strip=True))

    return info


def _extract_cell_data(cell: Tag) -> Dict[str, str]:
    """Extract data from a pricing cell, which might include multiple values."""
    # Initialize with the full text as default
    cell_text = _clean_text(cell.get_text(strip=True))

    # If the cell is just a simple text cell, return it directly
    if "-" in cell_text and len(cell_text) < 3:
        return cell_text

    # Check for price value and unit
    price_div = cell.find("div", class_="text-right flex-1")
    unit_div = cell.find("div", class_="text-xs text-gray-500 text-nowrap text-right")

    # If we have both components, organize them properly
    if price_div and unit_div:
        price = _clean_text(price_div.get_text(strip=True))
        unit = _clean_text(unit_div.get_text(strip=True))

        # If the price is empty, just return the text
        if not price.strip():
            return cell_text

        # Return structured data
        return {"price": price, "unit": unit}

    # Return the full text if we couldn't break it down
    return cell_text


def _clean_text(text: str) -> str:
    """Clean up text by removing extra whitespace and newlines."""
    # Replace multiple whitespace with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove any leading/trailing whitespace
    return text.strip()


def extract_pricing_data(html_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Main function to extract pricing data from HTML content.

    Args:
        html_content: HTML content of the pricing page

    Returns:
        Dictionary with API categories as keys and pricing information as values

    Example:


    """
    return parse_pricing_page(html_content)
