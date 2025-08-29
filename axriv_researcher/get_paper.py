import arxiv
import requests
import io
import re
from pypdf import PdfReader
from typing import Dict, Union

# It's good practice to have a single, reusable client instance
arxiv_client = arxiv.Client()

def _extract_text_from_paper(paper: arxiv.Result) -> str:
    """
    Private helper function to download a PDF from an arxiv.Result object,
    extract its text, and perform basic cleaning.
    """
    try:
        print(f"-> Fetching PDF from: {paper.pdf_url}")
        response = requests.get(paper.pdf_url)
        response.raise_for_status()  # Ensure the download was successful

        # Use an in-memory binary stream instead of saving to disk
        pdf_in_memory = io.BytesIO(response.content)
        
        # Read the PDF from memory
        reader = PdfReader(pdf_in_memory)
        print(f"-> The document has {len(reader.pages)} pages.")
        
        full_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        if not full_text:
            return f"Error: Could not extract any text from the PDF for '{paper.title}'. The PDF might be image-based or corrupted."

        # Basic post-processing to improve text for an LLM
        # 1. Join hyphenated words split across lines
        cleaned_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text)
        # 2. Consolidate paragraphs by removing single newlines
        cleaned_text = re.sub(r'(?<!\n)\n(?!\n)', ' ', cleaned_text)
        # 3. Normalize whitespace
        cleaned_text = re.sub(r' +', ' ', cleaned_text)
        
        return cleaned_text.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to download the PDF for '{paper.title}'. Reason: {e}"
    except Exception as e:
        # This can catch pypdf errors for corrupted or encrypted PDFs
        return f"Error: Failed to read or process the PDF for '{paper.al}'. Reason: {e}"


def get_arxiv_paper_text(
    query: str,
    search_field: str = "keyword"
) -> Union[Dict[str, str], str]:
    """
    A comprehensive tool to find an arXiv paper and extract its full text.

    This function searches for a paper on arXiv using various fields,
    downloads the PDF into memory, extracts the text, cleans it, and
    returns a structured dictionary of the paper's content.

    Args:
        query (str): The search term, which can be an ID, URL, title, etc.
        search_field (str, optional): The field to search against. Defaults to "keyword".
            Valid options are:
            - 'id': A specific arXiv paper ID (e.g., '2503.10150v2').
            - 'url': The full URL to the paper's abstract page.
            - 'title': Search for the paper by its title.
            - 'author': Search for papers by a specific author.
            - 'abstract': Search within the paper's abstract.
            - 'category': Search by subject category (e.g., 'cs.CL').
            - 'keyword': A general search across all fields (default).

    Returns:
        Union[Dict[str, str], str]: 
        - On success, a dictionary containing the paper's metadata and full text:
          {'title': str, 'id': str, 'url': str, 'summary': str, 'text': str}
        - On failure, an error message string.
    """
    search_field = search_field.lower()
    search_params = {
        "max_results": 1, # We only want the top result
        "sort_by": arxiv.SortCriterion.Relevance
    }

    if search_field == "id":
        search_params["id_list"] = [query]
    elif search_field == "url":
        # Regex to robustly parse arXiv ID from various URL formats
        match = re.search(r'(\d{4}\.\d{4,}(v\d+)?)', query)
        if not match:
            return f"Error: Could not parse a valid arXiv ID from the URL: {query}"
        arxiv_id = match.group(1)
        print(f"-> Parsed arXiv ID '{arxiv_id}' from URL.")
        search_params["id_list"] = [arxiv_id]
    elif search_field in ["title", "author", "abstract", "category", "keyword"]:
        # Use arXiv's advanced query syntax for targeted searches
        prefix_map = {
            "title": "ti", "author": "au", "abstract": "abs", "category": "cat"
        }
        prefix = prefix_map.get(search_field)
        # For titles and authors, quoting the query often yields better results
        if prefix:
            search_params["query"] = f'{prefix}:"{query}"'
        else: # For keyword search
            search_params["query"] = query
    else:
        return (f"Error: Invalid search_field '{search_field}'. Must be one of: "
                "'id', 'url', 'title', 'author', 'abstract', 'category', 'keyword'.")

    # Perform the search
    try:
        search = arxiv.Search(**search_params)
        print(f"-> Searching arXiv with parameters: {search_params}")
        paper = next(arxiv_client.results(search))
        
        print(f"-> Found paper: '{paper.title}' (ID: {paper.get_short_id()})")
        
        # Call the helper to get the text
        extracted_text = _extract_text_from_paper(paper)

        if extracted_text.startswith("Error:"):
            return extracted_text  # Return the error message from the helper

        # On success, return the full structured data
        return {
            "title": paper.title,
            "id": paper.get_short_id(),
            "url": paper.entry_id,
            "summary": paper.summary.replace('\n', ' '), # Clean summary newlines
            "text": extracted_text
        }

    except StopIteration:
        return f"Error: No papers found for your query."
    except Exception as e:
        return f"An unexpected error occurred during the arXiv search: {e}"

# --- EXAMPLES OF HOW TO USE THE TOOL ---
if __name__ == '__main__':
    print("--- Example 1: Searching by arXiv ID ---")
    result_by_id = get_arxiv_paper_text("2404.16130", search_field="id")
    if isinstance(result_by_id, dict):
        print(f"Title: {result_by_id['title']}")
        print(f"Summary: {result_by_id['summary'][:200]}...")
        print(f"Extracted Text Preview: {result_by_id['text'][:400]}...")
    else:
        print(result_by_id) # Print error message

    print("\n" + "="*50 + "\n")

    print("--- Example 2: Searching by Title ---")
    title_query = "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
    result_by_title = get_arxiv_paper_text(title_query, search_field="title")
    if isinstance(result_by_title, dict):
        print(f"Successfully found paper with ID: {result_by_title['id']}")
        print(f"Text length: {len(result_by_title['text'])} characters")
    else:
        print(result_by_title)

    print("\n" + "="*50 + "\n")
    
    print("--- Example 3: Searching by URL ---")
    url_query = "http://arxiv.org/abs/2401.18059v1"
    result_by_url = get_arxiv_paper_text(url_query, search_field="url")
    if isinstance(result_by_url, dict):
        print(f"Title: {result_by_url['title']}")
    else:
        print(result_by_url)

    print("\n" + "="*50 + "\n")

    print("--- Example 4: Handling a Not Found Error ---")
    not_found_result = get_arxiv_paper_text("This is not a real paper title", search_field="title")
    print(not_found_result)