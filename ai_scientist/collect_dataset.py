import csv
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Optional, Set, Union
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from htmldate import find_date

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# Global variables
driver = None


class ExtractedDataList(BaseModel):
    """List of data which were extracted from the text."""

    data: List[str] = Field(
        description="List of data which were extracted from the text."
    )


def generate_templates(data_collection_goal: str, llm: ChatAnthropic) -> List[str]:
    template_generation_prompt = """
    You are an expert at creating targeted, templated search queries for data collection. Your task is to generate 1-3 Google search query templates based on the user's data collection needs. These templates should focus on gathering relevant URLs for further analysis, balancing specificity with breadth.

    Guidelines:
    1. Analyze the data collection needs carefully, focusing on the core topic and key aspects to be gathered.
    2. Identify essential variables that could change in the search query (e.g., main keywords, locations).
    3. Do not include time-related placeholders (e.g., {{year}}).
    4. Do not include language-related placeholders (e.g., {{language}}).
    5. Determine if any variables are related pairs needing indexed placeholders. Use indexed placeholders for pairs of related information, e.g., {{variable[0]}} for the first item and {{variable[1]}} for the second item in the pair.
    6. Consider if the minus (-) operator is necessary to exclude any irrelevant terms. Use it only when clearly beneficial for data collection.
    7. Decide on the most effective order of elements in the template to ensure relevance to the data collection needs.
    8. Create 1-3 templates replacing variables with {{curly brace}} placeholders.
    9. Ensure the templates are specific enough to yield relevant results, but broad enough to capture a variety of sources for comprehensive data collection.
    10. Include essential keywords and phrases from the original request that define the topic and data to be collected, and that are not already covered by the placeholders.
    11. Do NOT use any Google search operators except the minus (-) operator. (No OR, AND, site:, etc.)
    12. Use simple keywords and phrases separated by spaces, with the occasional minus operator if needed.
    13. Reuse placeholders across templates when possible to maintain consistency.
    14. Generate multiple variations of the template to cover different possible phrasings or synonyms.
    15. If applicable, create templates for different aspects of the research question to ensure comprehensive coverage.

    Remember: Focus on creating templates that will gather a diverse set of relevant data sources, not on how the data will be analyzed later.

    Data collection goal: {data_collection_goal}

    Return ONLY the templates as a list of strings. DO NOT return anything other than the list of template strings.
    """

    prompt = PromptTemplate(
        template=template_generation_prompt, input_variables=["data_collection_goal"]
    )

    response = llm.invoke(prompt.format(data_collection_goal=data_collection_goal))

    # Extract the list of templates from the response
    templates = eval(response.content)
    return templates


def extract_placeholders(templates: List[str]) -> List[str]:
    """
    Extracts placeholders from templates.
    """
    placeholders = set()
    for template in templates:
        matches = re.findall(r"\{(\w+)(?:\[\d+\])?\}", template)
        placeholders.update(matches)
    return list(placeholders)


def generate_values(
    data_collection_goal: str,
    templates: List[str],
    placeholders: List[str],
    llm: ChatAnthropic,
) -> Dict[str, List[str]]:
    """
    Generates values for placeholders.
    """
    value_generation_prompt = """
    You are an expert at refining search queries. Generate relevant values for placeholders in the given search templates based on the provided context.

    Guidelines:
    1. Analyze the user input and search templates carefully.
    2. For each placeholder, generate a list of relevant values that is neither too short nor too long, but precisely what's needed to cover the scope of the query effectively.
    3. If placeholder values should be used together, generate a list of tuples of values.

    Data collection goal: {data_collection_goal}
    Search templates:
    {templates}
    Placeholders needing values: {placeholders}

    Return ONLY the values for each placeholder as a dictionary mapping placeholder names to lists of values. DO NOT return anything other than the dictionary of values.
    """

    prompt = PromptTemplate(
        template=value_generation_prompt,
        input_variables=["data_collection_goal", "templates", "placeholders"],
    )

    response = llm.invoke(
        prompt.format(
            data_collection_goal=data_collection_goal,
            templates=templates,
            placeholders=placeholders,
        )
    )

    # Extract the dictionary of values from the response
    values = eval(response.content)
    return values


def google_search_no_api(
    query: str, start_date: Optional[str], end_date: Optional[str]
) -> List[str]:
    """
    Searches for URLs using a query template.
    """
    global driver

    if start_date and end_date:
        date_range = f"after:{start_date} before:{end_date}"
        search_query = f"{query} {date_range}"
    else:
        search_query = query

    if driver is None:
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--geolocation=New York")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

    try:
        driver.get(
            "https://www.google.com/search?q=" + requests.utils.quote(search_query)
        )
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "search"))
        )
        search_results = driver.find_elements(By.CSS_SELECTOR, "div.yuRUbf")
        all_results = [
            result.find_element(By.TAG_NAME, "a").get_attribute("href")
            for result in search_results
        ]
        return all_results
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def search_and_save_urls(
    query_template: str,
    column_values: Dict[str, List[Union[str, List[str]]]],
    output_path: str,
    start_date: Optional[str],
    end_date: Optional[str],
    collect_dates: bool,
):
    """
    Searches for URLs using a query template and saves them to a CSV file.
    """
    urls_no_api = defaultdict(set)
    columns = []

    # Prepare column names, handling both single values and tuples
    for key, values in column_values.items():
        if isinstance(values[0], tuple):
            columns.extend([f"{key}_{i}" for i in range(len(values[0]))])
        else:
            columns.append(key)

    # Generate search queries and collect URLs
    for values in product(*column_values.values()):
        value_dict = {}
        for key, value in zip(column_values.keys(), values):
            if isinstance(value, tuple):
                value_dict[key] = value
            else:
                value_dict[key] = value

        query = query_template.format(**value_dict)
        results = google_search_no_api(query, start_date, end_date)
        key = tuple(values)
        urls_no_api[key].update(results)

    # Write results to CSV file
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = columns + ["URL"]
        if collect_dates:
            header.append("Date")
        writer.writerow(header)

        for key, urls in urls_no_api.items():
            for url in urls:
                row = []
                for item in key:
                    if isinstance(item, tuple):
                        row.extend(item)
                    else:
                        row.append(item)
                row.append(url)

                # Collect and add date if required
                if collect_dates:
                    try:
                        date = find_date(url)
                    except Exception:
                        date = None
                    row.append(date)

                writer.writerow(row)


def extract_text_from_url(url: str) -> str:
    """
    Extracts text from a URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Referer": "https://www.google.com",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return ""
    except requests.exceptions.ConnectionError:
        print("Connection error")
        return ""
    if response.status_code != 200:
        return ""
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


def extract_data_from_url(
    url: str, extraction_prompt: str, llm: ChatAnthropic
) -> ExtractedDataList:
    """
    Extracts data from a URL using a prompt and an LLM.
    """
    structured_llm = llm.with_structured_output(ExtractedDataList)
    text = extract_text_from_url(url)
    response = structured_llm.invoke(f"{extraction_prompt}\n\nText: {text}")
    return response


def smart_scraper(
    urls: List[str], prompt: str, llm: ChatAnthropic
) -> List[Optional[Dict[str, str]]]:
    """
    Scraper that extracts data from URLs using a prompt and an LLM.
    """
    results = []
    for i, url in enumerate(urls):
        try:
            extracted_data = extract_data_from_url(url, prompt, llm)
            result = {f"data{i+1}": data for i, data in enumerate(extracted_data.data)}
            results.append(result)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            results.append(None)
    return results


def collect_dataset(prompt: str, folder_name: str, model):
    llm = model

    # Generate search templates
    templates = generate_templates(prompt, llm)

    # Extract placeholders from templates
    placeholders = extract_placeholders(templates)

    # Generate values for placeholders
    column_values = generate_values(prompt, templates, placeholders, llm)

    # Perform search and save URLs
    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, suffix=".csv"
    ) as temp_file:
        output_path = temp_file.name

    for template in templates:
        search_and_save_urls(template, column_values, output_path, None, None, True)

    # Read search results
    df = pd.read_csv(output_path)

    # Perform smart scraping
    urls = df["URL"].tolist()
    scraping_results = smart_scraper(urls, prompt, llm)

    # Process scraped results
    for i, result in enumerate(scraping_results):
        if result:
            for key, value in result.items():
                df.loc[i, f"Scraped_{key}"] = value

    # Clean up temporary file
    os.unlink(output_path)

    df.to_csv(f"{folder_name}/dataset.csv", index=False)
