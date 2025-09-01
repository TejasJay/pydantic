# research_assistant.py

# --- Imports ---
import os
import re
import asyncio
from dataclasses import dataclass
from typing import Dict, Union, List, Optional

import arxiv
from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, RunContext
from pypdf import PdfReader
import gradio as gr

# --- Configuration ---
load_dotenv()
MODEL_NAME = os.getenv("PYDANTIC_AI_MODEL", "google-gla:gemini-1.5-pro-latest")

# --- Pydantic Models for Structured Output ---
class PaperAnalysis(BaseModel):
    title: str = Field(description="The full title of the paper.")
    authors: List[str] = Field(description="A list of the paper's authors.")
    summary: str = Field(description="The original abstract/summary of the paper.")
    introduction: str = Field(description="A concise, one-paragraph summary of the paper's introduction, in your own words.")
    methodology: str = Field(description="A detailed explanation of the paper's methodology, techniques, and core ideas.")
    conclusion: str = Field(description="A summary of the paper's results and conclusions.")

class MathAnalysis(BaseModel):
    equation: str = Field(description="A key equation or mathematical formula found in the paper, written in LaTeX format.")
    explanation: str = Field(description="A detailed, technical explanation of what the equation does and its components.")
    simplified_explanation: str = Field(description="A simplified, intuitive explanation of the equation, suitable for a beginner.")
    real_world_example: str = Field(description="A practical, real-world scenario or use case where this equation is applied.")

class CodeAnalysis(BaseModel):
    required_dependencies: List[str] = Field(description="A list of Python libraries that must be pip installed to run the code (e.g., ['numpy', 'scikit-learn']).")
    python_code: str = Field(description="A complete, self-contained, and runnable Python script that demonstrates the core concept of the paper.")
    explanation: str = Field(description="A detailed explanation of how the Python code implements the paper's methodology and concepts.")
    improvement_suggestion: str = Field(description="A suggestion on how the code could be improved, made more efficient, or written in a more modern style.")

class FullReport(BaseModel):
    paper_summary: PaperAnalysis
    math_analysis: List[MathAnalysis]
    code_analysis: Optional[CodeAnalysis] = Field(default=None)
    final_takeaways: str

# --- Dependency Management ---
@dataclass
class ArxivDependencies:
    arxiv_client: arxiv.Client

# --- Tool Definition ---

# --- FIX HIGHLIGHT: Create a synchronous helper function ---
# Logic: We encapsulate all the blocking I/O calls (arxiv search, download, PDF parsing)
# into a single, synchronous helper function.
def _sync_get_arxiv_paper_text(
    client: arxiv.Client,
    query: str,
    search_field: str
) -> Union[Dict[str, Union[str, List[str]]], str]:
    """Synchronous helper to perform the blocking search and download operations."""
    search_params = {"max_results": 1, "sort_by": arxiv.SortCriterion.Relevance}
    search_field = search_field.lower()

    if search_field == "id":
        search_params["id_list"] = [query]
    elif search_field == "url":
        match = re.search(r'(\d{4}\.\d{4,}(v\d+)?)', query)
        if not match: return f"Error: Could not parse a valid arXiv ID from the URL: {query}"
        search_params["id_list"] = [match.group(1)]
    elif search_field in ["title", "author", "abstract", "category", "keyword"]:
        prefix_map = {"title": "ti", "author": "au", "abstract": "abs", "category": "cat"}
        prefix = prefix_map.get(search_field)
        search_params["query"] = f'{prefix}:"{query}"' if prefix else query
    else:
        return f"Error: Invalid search_field '{search_field}'."

    try:
        print(f"-> Searching arXiv with parameters: {search_params}")
        search = arxiv.Search(**search_params)
        paper = next(client.results(search))
        
        print(f"-> Found paper: '{paper.title}' (ID: {paper.get_short_id()})")
        
        # This is a blocking network call
        pdf_path = paper.download_pdf()
        text = ""
        # This is a blocking file I/O and CPU-bound call
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        os.remove(pdf_path)

        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r' +', ' ', text).strip()

        return {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary.replace('\n', ' '),
            "text": text
        }
    except StopIteration:
        return f"Error: No papers found for your query '{query}' with search_field '{search_field}'."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

async def get_arxiv_paper_text(
    ctx: RunContext[ArxivDependencies],
    query: str,
    search_field: str = "keyword"
) -> Union[Dict[str, Union[str, List[str]]], str]:
    """
    Asynchronous wrapper for the arXiv tool. It runs the blocking search
    and download operations in a separate thread to avoid freezing the event loop.
    """
    # --- FIX HIGHLIGHT: Run the blocking code in a separate thread ---
    # Logic: `asyncio.to_thread` tells the event loop to run our synchronous helper function
    # in a background thread pool. This frees up the main thread, allowing Gradio's UI
    # to remain responsive and process progress bar updates.
    return await asyncio.to_thread(
        _sync_get_arxiv_paper_text, ctx.deps.arxiv_client, query, search_field
    )

# --- Agent Definitions ---
# Logic: We define four specialized agents, each with a distinct role and a detailed set of instructions.
# This multi-agent approach breaks down a complex task into smaller, more manageable steps,
# leading to higher quality and more reliable results.

summary_agent = Agent(
    model=MODEL_NAME,
    deps_type=ArxivDependencies,
    output_type=PaperAnalysis,
    instructions="""
    You are a senior research analyst. Your primary goal is to find a specific research paper on arXiv and produce a high-quality, structured summary of its contents.

    Your process MUST be as follows:
    1.  **Find the Paper**: Use the `get_arxiv_paper_text` tool with the user's query. This tool will provide you with the paper's title, authors, abstract, and full text.
    2.  **Analyze the Content**: Carefully read all the information returned by the tool.
    3.  **Populate the Summary**: Synthesize your findings to populate ALL fields of the `PaperAnalysis` output model.
        -   **`title`**: Use the exact title returned by the tool.
        -   **`authors`**: List all authors returned by the tool.
        -   **`summary`**: Use the exact abstract returned by the tool.
        -   **`introduction`**: Read the beginning of the full text and write a new, concise paragraph in your own words that summarizes the paper's introduction, problem statement, and goals.
        -   **`methodology`**: Read the core sections of the paper and write a detailed but clear explanation of the techniques, algorithms, and architecture used. This is the most important part.
        -   **`conclusion`**: Read the end of the paper and summarize the key results, findings, and future work.
    4.  **Final Output**: Your final response must be ONLY the structured `PaperAnalysis` JSON object.
    """
)

math_agent = Agent(
    model=MODEL_NAME,
    deps_type=ArxivDependencies,
    output_type=List[MathAnalysis],
    instructions="""
    You are an expert mathematician and data scientist. Your task is to analyze the full text of a research paper to identify, extract, and explain the most significant mathematical concepts.

    Your process MUST be as follows:
    1.  **Scan the Text**: Read the entire paper text provided in the prompt.
    2.  **Identify Key Equations**: Look for the 1-3 most important and central mathematical formulas or equations. Do not extract simple variables or trivial equations.
    3.  **Create Analysis Objects**: For each significant equation you identify, create a `MathAnalysis` object and populate all of its fields:
        -   **`equation`**: The mathematical formula, precisely transcribed in LaTeX format.
        -   **`explanation`**: A detailed, technical breakdown of the equation, explaining each variable and the overall mathematical operation.
        -   **`simplified_explanation`**: An intuitive, simplified explanation of the equation's purpose, as if you were explaining it to a student.
        -   **`real_world_example`**: A practical example of how this mathematical concept is or could be used.
    4.  **Final Output**: Return a JSON list containing all the `MathAnalysis` objects you created. If you find no significant mathematical content in the paper, you MUST return an empty list `[]`.
    """
)

code_agent = Agent(
    model=MODEL_NAME,
    deps_type=ArxivDependencies,
    output_type=CodeAnalysis,
    instructions="""
    You are a senior software engineer and Python expert. Your goal is to read the full text of a research paper and generate a simple, runnable Python script that serves as a practical demonstration of the paper's core concepts.

    Your process MUST be as follows:
    1.  **Understand the Concept**: Read the provided paper text to deeply understand its main algorithm, technique, or methodology.
    2.  **Generate Code**: Write a clear, self-contained, and runnable Python script that implements a simplified version of this core concept. The code should be well-commented to explain each step.
    3.  **List Dependencies**: Identify all the external Python libraries required to run your script (e.g., numpy, torch, scikit-learn).
    4.  **Explain the Implementation**: Write a clear explanation of how your Python code translates the paper's theoretical ideas into a practical example.
    5.  **Suggest Improvements**: Provide a thoughtful suggestion on how the code could be improved, made more efficient, or extended.
    6.  **Populate the Output**: Fill all fields of the `CodeAnalysis` output model. If the paper is purely theoretical and cannot be reasonably implemented in a simple script, you must still produce a valid `CodeAnalysis` object, but you can state this limitation in the `explanation` and `improvement_suggestion` fields and provide an empty list for `required_dependencies` and a placeholder script (e.g., `print('This concept is theoretical and not directly implementable.')`).
    """
)

full_agent = Agent(
    model=MODEL_NAME,
    output_type=FullReport,
    instructions="""
    You are a meticulous report editor. Your task is to assemble a final, comprehensive report from the structured JSON analyses provided in the prompt.

    Your process MUST be as follows:
    1.  **Assemble the Report**: Take the `PaperSummary`, `Mathematical Analysis`, and `Code Analysis` JSON objects provided in the prompt and place them directly into the corresponding fields of the `FullReport` output model. Do not alter or summarize their content.
    2.  **Synthesize Takeaways**: After assembling the report, read through all the provided analyses. Write a new, insightful paragraph for the `final_takeaways` field that summarizes the paper's overall significance, its key innovations, and its potential impact.
    3.  **Final Output**: Your final response must be the complete, structured `FullReport` JSON object.
    """
)

summary_agent.tool(get_arxiv_paper_text)
math_agent.tool(get_arxiv_paper_text)

# --- Main Orchestration Function ---
async def generate_full_report(user_prompt: str, progress=gr.Progress(track_tqdm=True)):
    # (No changes needed here, this logic is correct)
    arxiv_client = arxiv.Client()
    dependencies = ArxivDependencies(arxiv_client=arxiv_client)

    progress(0.1, desc="Stage 1/4: Finding and summarizing the paper...")
    summary_result = await summary_agent.run(user_prompt, deps=dependencies)
    
    if not isinstance(summary_result.output, PaperAnalysis):
        raise gr.Error(f"Failed to retrieve or summarize the paper: {summary_result.output}")
    
    paper_summary = summary_result.output
    
    paper_text = ""
    for message in summary_result.new_messages():
        for part in message.parts:
            if part.part_kind == 'tool-return' and part.tool_name == 'get_arxiv_paper_text':
                if isinstance(part.content, dict):
                    paper_text = part.content.get("text", "")
                    break
        if paper_text: break
    
    if not paper_text:
        raise gr.Error("Could not retrieve full paper text from the summary agent's run.")

    progress(0.4, desc="Stage 2/4: Analyzing mathematical concepts...")
    math_prompt = f"Analyze the following paper text for mathematical concepts:\n\n{paper_text}"
    math_result = await math_agent.run(math_prompt, deps=dependencies)
    math_analysis = math_result.output

    progress(0.7, desc="Stage 3/4: Generating Python code implementation...")
    code_prompt = f"Based on the methodology in the following paper text, create a simple Python implementation:\n\n{paper_text}"
    code_result = await code_agent.run(code_prompt, deps=dependencies)
    code_analysis = code_result.output

    progress(0.9, desc="Stage 4/4: Compiling the final report...")
    math_adapter = TypeAdapter(List[MathAnalysis])
    math_json = math_adapter.dump_json(math_analysis, indent=2) if isinstance(math_analysis, list) else b'[]'
    code_json = code_analysis.model_dump_json(indent=2) if isinstance(code_analysis, CodeAnalysis) else 'null'

    final_prompt = f"""
    Please assemble the following analyses into a single, complete FullReport.
    Paper Summary: {paper_summary.model_dump_json(indent=2)}
    Mathematical Analysis: {math_json.decode()}
    Code Analysis: {code_json}
    """
    
    full_report_result = await full_agent.run(final_prompt)
    
    if not isinstance(full_report_result.output, FullReport):
        raise gr.Error(f"Final agent failed to produce a valid report: {full_report_result.output}")

    return full_report_result.output

# This block allows the script to be run from the command line for testing.
if __name__ == '__main__':
    async def test_run():
        test_prompt = "Find the paper with ID '1706.03762' (Attention Is All You Need) and provide a full report on it."
        class DummyProgress:
            def __init__(self, track_tqdm=False): pass
            def __call__(self, progress, desc=""): print(desc)
        
        report = await generate_full_report(test_prompt, progress=DummyProgress())
        print("\n" + "="*20 + " FINAL COMPREHENSIVE REPORT " + "="*20)
        print(report.model_dump_json(indent=2))
    
    asyncio.run(test_run())