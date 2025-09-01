# app.py

import gradio as gr
import asyncio
from typing import Optional
# Import the core logic from your research_assistant library
from research_assistant import generate_full_report, FullReport

def display_report(report: Optional[FullReport]):
    """
    Takes a FullReport object and returns a list of Gradio component updates.
    This function is called after the main logic is complete.
    """
    if not report:
        error_md = gr.Markdown("### Report generation failed. Please check the console for errors.", visible=True)
        return [error_md] + [gr.update(visible=False)] * 4

    # --- Paper Summary Section ---
    summary_md = f"""
    # {report.paper_summary.title}
    **Authors:** {', '.join(report.paper_summary.authors)}
    
    ## Abstract
    {report.paper_summary.summary}
    
    ## Introduction
    {report.paper_summary.introduction}
    
    ## Methodology
    {report.paper_summary.methodology}
    
    ## Conclusion
    {report.paper_summary.conclusion}
    """

    # --- Math Analysis Section (Dynamic Content Generation) ---
    if report.math_analysis:
        math_md_parts = ["# Mathematical Concepts"]
        for i, math in enumerate(report.math_analysis):
            math_md_parts.append(f"""
            ### Concept {i+1}: Equation
            $${math.equation}$$
            **Technical Explanation:**
            {math.explanation}
            **Simplified Explanation:**
            {math.simplified_explanation}
            **Real-World Example:**
            {math.real_world_example}
            <hr>
            """)
        math_md = "\n".join(math_md_parts)
        math_update = gr.Markdown(math_md, visible=True)
    else:
        math_update = gr.Markdown("### No significant mathematical concepts were identified.", visible=True)

    # --- Code Analysis Section (Dynamic Content Generation) ---
    if report.code_analysis and report.code_analysis.python_code:
        code_md = f"""
        # Code Implementation
        ### Required Dependencies
        To run the code below, first install the following libraries:
        ```bash
        pip install {' '.join(report.code_analysis.required_dependencies)}
        ```
        ### Code Explanation
        {report.code_analysis.explanation}
        
        ### Improvement Suggestion
        {report.code_analysis.improvement_suggestion}
        """
        code_explanation_update = gr.Markdown(code_md, visible=True)
        code_block_update = gr.Code(
            value=report.code_analysis.python_code,
            language="python",
            label="Generated Python Implementation",
            visible=True
        )
    else:
        code_explanation_update = gr.Markdown("### No implementable code was generated for this paper.", visible=True)
        code_block_update = gr.Code(visible=False)

    # --- Final Takeaways Section ---
    takeaways_md = f"""
    # Final Takeaways
    {report.final_takeaways}
    """

    return [
        gr.Markdown(summary_md, visible=True),
        math_update,
        code_explanation_update,
        code_block_update,
        gr.Markdown(takeaways_md, visible=True)
    ]

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Research Assistant") as demo:
    gr.Markdown("# AI Research Assistant")
    gr.Markdown("Enter an arXiv paper ID, title, or keyword to generate a full, multi-agent report.")

    with gr.Row():
        query_input = gr.Textbox(
            label="Enter Paper ID, Title, or Keyword",
            placeholder="e.g., '1706.03762' or 'Attention Is All You Need'",
            scale=4,
        )
        submit_button = gr.Button("Generate Report", variant="primary", scale=1)

    with gr.Column(visible=False) as output_column:
        summary_output = gr.Markdown()
        math_output = gr.Markdown()
        code_explanation_output = gr.Markdown()
        code_output = gr.Code(language="python")
        takeaways_output = gr.Markdown()

    outputs = [
        summary_output,
        math_output,
        code_explanation_output,
        code_output,
        takeaways_output
    ]

    # --- FIX HIGHLIGHT: Make the wrapper function async ---
    # Logic: By making this function `async def`, we allow Gradio's event loop to manage it.
    # This enables the `progress` object to send updates back to the UI in real-time
    # while the `await generate_full_report` is running.
    async def wrapper_generate_report(query, progress=gr.Progress(track_tqdm=True)):
        """
        An async generator function that orchestrates the UI updates.
        """
        initial_updates = [gr.Column(visible=True)] + [gr.update(visible=False)] * 5
        yield initial_updates
        
        try:
            # Use `await` instead of `asyncio.run()`. This is the key change.
            report = await generate_full_report(query, progress)
            final_content_updates = display_report(report)
        except Exception as e:
            print(f"An error occurred during report generation: {e}")
            raise gr.Error(str(e))

        final_updates = [gr.skip()] + final_content_updates
        yield final_updates

    submit_button.click(
        fn=wrapper_generate_report,
        inputs=[query_input],
        outputs=[output_column] + outputs
    )

if __name__ == "__main__":
    demo.launch()