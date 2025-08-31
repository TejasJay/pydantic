import os
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import yt_dlp

# Import the function from your other module
from video_to_pdf_generator import generate_video_notes
from youtube_transcript_api import YouTubeTranscriptApi


# --- Define the Pydantic Output Structures ---
class VideoSearchResult(BaseModel):
    """Represents a single YouTube video search result."""
    title: str = Field(description="The official title of the YouTube video.")
    url: str = Field(description="The full, direct URL to the YouTube video.")


class TextSummaryOutput(BaseModel):
    """The output from the YouTube processing agent."""
    title: str = Field(description="A concise and engaging title for the summary.")
    subject: str = Field(description="The main subject or topic of the video.")
    explanation: str = Field(description="A very detailed, non truncated, paragraph-based explanation of the video transcript.")
    # full_text: str = Field(description="The complete non-truncated text extracted from the transcript")


class FinalAgentOutput(BaseModel):
    """The final, combined output from the YouTube processing agent."""
    text_summary: Optional[TextSummaryOutput] = Field(default=None, description="The detailed text-based summary and analysis of the video transcript.")
    error: str | None = Field(description="Explanation for not generating the result")

# --- Initialize the Agent ---
agent = Agent(
    model='google-gla:gemini-1.5-pro',
    instructions=(
        "You are an intelligent YouTube video analysis assistant. Your workflow is precise and must be followed in order:\n\n"
        "1. **Fetch Transcript:** Your first and mandatory action is to use the `get_youtube_transcript` tool with the provided video URL to get the full text.\n\n"
        "2. **Analyze and Decide:** Carefully analyze the entire transcript. Your goal is to determine if the video is visually complex enough to warrant a PDF report. You MUST call the `create_pdf_report_from_video` tool ONLY IF the transcript contains strong evidence of visual aids, such as keywords like 'code', 'diagram', 'slide', 'presentation', 'schema', 'query,' or phrases implying a visual explanation (e.g., 'as you can see on the screen,' 'this diagram shows'). If the video seems to be primarily dialogue, DO NOT call the PDF tool.\n\n"
        "3. **Generate Final Output:** After making your decision, generate the detailed text-based summary from the transcript. Assemble the final output structure. The `pdf_report_path` field must contain the path returned by the tool if you ran it in Step 2; otherwise, it must be null.\n\n"
        "4. **Error Handling:** If any step fails (e.g., the transcript cannot be fetched), populate the `error_message` field and leave the other fields null."
    ),
    output_type=FinalAgentOutput,
)

# --- Define the Agent's Tools ---
@agent.tool_plain()
def get_youtube_transcript(video_url: str) -> str:
    """
    Gets the full text transcript from a YouTube video using its URL.

    Args:
        video_url: The full YouTube video URL.
    """
    print(f"Fetching transcript for verified URL: {video_url}")
    try:
        regex_pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com|youtu\.be)\/(?:watch\?v=|embed\/|shorts\/)?([\w-]{11})"
        video_id_match = re.search(regex_pattern, video_url)
        video_id = video_id_match.group(1) if video_id_match else None

        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)
        fetched_list = [i.text for i in fetched_transcript]
        full_text = " ".join(fetched_list)
        return full_text
    except Exception as e:
        return f"Error fetching transcript: {e}"


@agent.tool_plain()
def create_pdf_report_from_video(video_url: str) -> str:
    """
    Processes a visually complex YouTube video to extract keyframes and transcript sections
    and compiles them into a professional PDF report.
    **Only use this tool if the transcript indicates the video contains diagrams, code, or slides.**

    Args:
        video_url: The full YouTube video URL to process.

    Returns:
        The absolute local filesystem path to the generated PDF file.
    """
    print(f"Starting PDF generation for visually complex video: {video_url}")
    output_directory = "./youtube_agent/generated_notes"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    pdf_path = generate_video_notes(video_url=video_url, output_dir=output_directory)
    return pdf_path


def agent_answer(user_query):
    print(f"Running agent for query: '{user_query}'")
    result = agent.run_sync(user_query)
    print('\n----- AGENT RESULTS -----')
    if result.output.text_summary:
        print("\n--- Text Summary ---")
        print(f"Title: {result.output.text_summary.title}")
        print(f"Subject: {result.output.text_summary.subject}")
    print('\n--- Complete Pydantic JSON Output ---')
    output_dict = result.output.model_dump()
    print(json.dumps(output_dict, indent=4))



# --- Main Execution Block ---
if __name__ == '__main__':
    agent_answer(user_query= "https://www.youtube.com/watch?v=T5rodl0FWRQ&t=308s")







