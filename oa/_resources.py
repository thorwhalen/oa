"""Tools to get resources for oa"""

import os
import dol

# -------------------------------------------------------------------------------------
# Stores

from dol import cache_this

from oa.util import data_files

dflt_resources_dir = str(data_files.parent.parent / 'misc' / 'data' / 'resources')


# -------------------------------------------------------------------------------------
# Resources class

from dataclasses import dataclass, field
import json
import pathlib
from functools import partial
from typing import Dict, Any, Optional, Callable, Union, List

import pandas as pd
import dol


dflt_pricing_url = 'https://platform.openai.com/docs/pricing'

@dataclass
class Resources:
    """
    Data Access class for OpenAI pricing information.

    This class manages the retrieval, computation, and caching of
    OpenAI pricing data through a chain of computations.

    >>> r = Resources()  # doctest: +ELLIPSIS
    >>> r.resources_dir  # doctest: +ELLIPSIS
    '...misc/data'
    """

    resources_dir: str = dflt_resources_dir
    pricing_url: str = dflt_pricing_url

    schema_description_key: str = 'openai_api_pricing_schema_description.txt'
    schema_key: str = 'api_pricing_schema.json'
    pricing_html_key: str = 'openai_api_pricing.html'
    pricing_info_key: str = 'openai_api_pricing_info.json'
    pricing_info_from_ai_key: str = 'openai_api_pricing_info_from_ai.json'

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
            from oa._resources import get_pricing_page_html

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

    @cache_this(cache='text_store', key=pricing_html_key)
    def pricing_page_html(self):
        """Retrieve the HTML for the OpenAI pricing page."""
        return get_pricing_page_html(self.pricing_url)

    @cache_this(cache='json_store', key=pricing_info_key)
    def pricing_info(self):
        return extract_pricing_data(self.pricing_page_html)

    @dol.cache_this(cache='text_store', key=lambda self: self.schema_description_key)
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

    @dol.cache_this(cache='json_store', key=lambda self: self.schema_key)
    def schema(self) -> Dict[str, Any]:
        """Generate or retrieve the schema for OpenAI API pricing."""
        return self.infer_schema_from_verbal_description(self.schema_description)

    @dol.cache_this(cache='json_store', key=lambda self: self.pricing_info_from_ai_key)
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
        tables = self.pricing_info_from_ai.get('OpenAI_API_Pricing_Schema', {})

        # Create a mapping interface that transforms table data to DataFrames
        return dol.add_ipython_key_completions(
            dol.wrap_kvs(
                tables, value_decoder=lambda x: pd.DataFrame(x.get('pricing_table', []))
            )
        )

    def list_pricing_categories(self) -> List[str]:
        """List available pricing categories."""
        return list(self.pricing_info_from_ai.get('OpenAI_API_Pricing_Schema', {}))

    def get_pricing_table(self, category: str) -> pd.DataFrame:
        """Get a specific pricing table as a pandas DataFrame."""
        return self.pricing_tables_from_ai[category]


# -------------------------------------------------------------------------------------
# Extract pricing data from HTML content


import re
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict


# TODO: Make it get the html from local cache if it exists
def get_pricing_page_html(url=dflt_pricing_url):
    """
    Get the HTML content of the pricing page.

    Need to use chrome to render the page and get the full content.
    Sometimes need to do it twice, since some catpcha pages are shown sometimes.
    """
    from tabled.html import url_to_html_func

    url_to_html = url_to_html_func(('chrome', dict(wait=10)))
    return url_to_html(url)


def parse_pricing_page(html_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the pricing page HTML and extract structured data about API pricing.

    Args:
        html_content: HTML content of the pricing page

    Returns:
        Dictionary with API categories as keys and pricing information as values
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    result = {}

    # Find all sections that contain pricing tables
    sections = soup.find_all('section')

    for section in sections:
        section_data = _parse_section(section)
        if section_data:
            category_name, data = section_data
            result[category_name] = data

    return result


def _parse_section(section: Tag) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parse a section containing pricing information.

    Args:
        section: BeautifulSoup Tag containing a section

    Returns:
        Tuple of (category_name, data_dict) or None if section has no pricing data
    """
    # Try to find the category name
    category_name_div = section.find('div', class_='font-medium')
    if not category_name_div:
        return None

    category_name = category_name_div.get_text(strip=True)

    # Initialize data dictionary for this category
    data = {"pricing_table_schema": [], "pricing_table": [], "extras": {}}

    # Extract any extra information
    extras = _extract_extras(section)
    if extras:
        data["extras"] = extras

    # Find tables in this section
    tables = section.find_all('table')

    for table_idx, table in enumerate(tables):
        # If there's more than one table, we'll assume the second is for batch pricing
        table_prefix = "" if table_idx == 0 else " - Batch"

        # Extract table schema (column headers)
        headers = table.find_all('th')
        schema = [header.get_text(strip=True) for header in headers]

        if table_idx == 0:
            data["pricing_table_schema"] = schema
        else:
            data[f"pricing_table_schema{table_prefix}"] = schema

        # Extract table rows
        rows = table.find_all('tr')[1:]  # Skip header row
        table_data = []

        for row in rows:
            row_data = {}
            cells = row.find_all('td')

            # Process cells
            for i, cell in enumerate(cells):
                if i < len(schema):  # Ensure we have a corresponding schema item
                    column_name = schema[i]

                    # Handle model name and alternate model names
                    if i == 0:  # First column is typically the model name
                        model_names = _extract_model_names(cell)
                        if len(model_names) > 1:
                            row_data["Model"] = model_names[0]
                            row_data["Alternate Model Names"] = model_names[1:]
                        elif len(model_names) == 1:
                            row_data["Model"] = model_names[0]
                    else:
                        # For other columns, extract the text content
                        row_data[column_name] = cell.get_text(strip=True)

            if row_data:
                table_data.append(row_data)

        if table_idx == 0:
            data["pricing_table"] = table_data
        else:
            data[f"pricing_table{table_prefix}"] = table_data

    return category_name, data


def _extract_extras(section: Tag) -> Dict[str, str]:
    """
    Extract extra information from a section.

    Args:
        section: BeautifulSoup Tag containing a section

    Returns:
        Dictionary of extra information
    """
    extras = {}

    # Look for price per information
    price_info = section.find(
        'div', class_='flex flex-1 items-center justify-end gap-1 text-xs text-gray-600'
    )
    if price_info:
        extras["price_info"] = price_info.get_text(strip=True)

    # Look for description text
    description = section.find('div', class_='text-sm text-gray-600')
    if description:
        extras["description"] = description.get_text(strip=True)

    return extras


def _extract_model_names(cell: Tag) -> List[str]:
    """
    Extract model names from a table cell.

    Args:
        cell: BeautifulSoup Tag containing a table cell

    Returns:
        List of model names
    """
    model_names = []

    # Look for main model name
    main_model = cell.find('div', class_='text-gray-900')
    if main_model:
        model_names.append(main_model.get_text(strip=True))

    # Look for alternate model names
    alt_models = cell.find_all(
        'div', class_='text-gray-600 flex items-center group text-nowrap'
    )
    for alt in alt_models:
        # Skip the bullet point and extract the model name
        model_text = alt.get_text(strip=True)
        # Remove the bullet character if present
        model_text = re.sub(r'^[•·]', '', model_text).strip()
        if model_text and model_text not in model_names:
            model_names.append(model_text)

    # Look for specific model versions
    model_versions = cell.find_all('div', class_='text-xs text-gray-600')
    for version in model_versions:
        version_text = version.get_text(strip=True)
        if version_text and version_text not in model_names:
            model_names.append(version_text)

    return model_names


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
