import cv2
import os
import yt_dlp
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers.util import cos_sim
import time
from datetime import datetime, timedelta
from youtube_transcript_api import YouTubeTranscriptApi
from fpdf import FPDF
import nltk
import re
import multiprocessing
import tempfile
import shutil

# --- One-time NLTK download ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' model for sentence tokenization...")
    nltk.download('punkt')
    print("Download complete.")

# --- Global Configuration & Tuning ---
MODEL_ID = "openai/clip-vit-base-patch32" 
BATCH_SIZE = 16 
FRAMES_TO_PROCESS_PER_SECOND = 2
SIMILARITY_THRESHOLD = 0.975 
KEYFRAME_COOLDOWN_SECONDS = 3
PARAGRAPH_PAUSE_THRESHOLD = 1.0
NUM_CHUNKS = os.cpu_count() or 8

# --- Helper Functions (Internal to the module) ---

def _format_timestamp(seconds: float) -> str:
    """Formats a duration in seconds into a clean H:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))

def _sanitize_filename(text: str, max_length: int = 50) -> str:
    """Cleans a string to make it a valid filename."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "", text)
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:max_length]

def _format_text_into_points(text: str) -> str:
    """Splits a block of text into bulleted sentences using NLTK."""
    if not text.strip(): return ""
    sentences = nltk.sent_tokenize(text)
    return "\n".join(f"- {s.strip()}" for s in sentences)

# --- Core Logic Functions ---

def _get_video_stream_url_and_info(video_url: str) -> tuple[dict, str]:
    """
    Gets video metadata and a direct streaming URL without downloading the file.

    Args:
        video_url (str): The URL of the YouTube video.

    Returns:
        tuple[dict, str]: A tuple containing the video's metadata dictionary
                          and a direct URL to the video stream.
    """
    print(f"Fetching video stream info for: {video_url}")
    ydl_opts = {
        'format': 'best[ext=mp4]/best', # Get the best MP4 stream
        'quiet': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=False)
        
        best_format = None
        for f in info_dict.get('formats', []):
            if f.get('ext') == 'mp4' and f.get('vcodec') != 'none':
                if best_format is None or f.get('height', 0) > best_format.get('height', 0):
                    best_format = f
        
        if not best_format or 'url' not in best_format:
            raise RuntimeError("Could not find a suitable MP4 stream URL for the video.")

    stream_url = best_format['url']
    print("Found video stream URL.")
    return info_dict, stream_url

def _get_transcript(video_id: str) -> list:
    """Fetches the full transcript for a given YouTube video ID."""
    print("Fetching transcript...")
    try:
        api = YouTubeTranscriptApi()
        fetched_transcript = api.fetch(video_id)
        return fetched_transcript.to_raw_data()
    except Exception as e:
        print(f"Could not retrieve transcript for video ID {video_id}: {e}")
        return []

def _identify_keyframes_parallel(stream_url: str, video_duration: float, temp_keyframes_path: str) -> list:
    """Identifies all visually distinct keyframes from a video stream using parallel processing."""
    chunk_duration = video_duration / NUM_CHUNKS
    tasks = []
    for i in range(NUM_CHUNKS):
        start_time = i * chunk_duration
        end_time = (i + 1) * chunk_duration
        tasks.append((i, stream_url, start_time, end_time, temp_keyframes_path))

    start_process_time = time.time()
    print(f"Starting parallel processing with {NUM_CHUNKS} chunks...")
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    with multiprocessing.Pool(processes=os.cpu_count(), initializer=_init_worker, initargs=(MODEL_ID, device_str)) as pool:
        results = pool.starmap(_process_chunk, tasks)

    all_keyframes = [item for sublist in results for item in sublist]
    all_keyframes.sort(key=lambda x: x['timestamp'])
    
    print(f"\nFinished parallel processing. Found {len(all_keyframes)} total keyframes in {time.time() - start_process_time:.2f} seconds.")
    return all_keyframes

def _associate_transcript_with_keyframes(raw_transcript: list, all_keyframes: list) -> list:
    """Groups the transcript into logical sections and associates them with the best keyframe."""
    print("Grouping transcript and associating with keyframes...")
    
    if not raw_transcript:
        return [{
            "image_path": kf['image_path'],
            "timestamp_str": _format_timestamp(kf['timestamp']),
            "transcript": ""
        } for kf in all_keyframes]

    paragraphs = _group_transcript_by_keyframes(raw_transcript, all_keyframes, PARAGRAPH_PAUSE_THRESHOLD)
    
    final_notes_data = []
    for paragraph in paragraphs:
        relevant_keyframes = [kf for kf in all_keyframes if kf['timestamp'] <= paragraph['start_time']]
        if relevant_keyframes:
            best_keyframe = relevant_keyframes[-1]
            final_notes_data.append({
                "image_path": best_keyframe['image_path'],
                "timestamp_str": _format_timestamp(paragraph['start_time']),
                "transcript": _format_text_into_points(paragraph['text']),
            })
    return final_notes_data

def _create_pdf_report(notes_data: list, info_dict: dict, video_url: str, output_pdf_path: str):
    """Generates the final, professional PDF report from the processed notes data."""
    print(f"Compiling professional PDF at: {output_pdf_path}")
    if not notes_data:
        print("No notes were generated. Cannot create PDF.")
        return
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    video_title = info_dict.get('title', 'Video Notes')
    pdf.multi_cell(0, 20, video_title, 0, "C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Generated from YouTube video:\n{video_url}")
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Total Sections Found: {len(notes_data)}", 0, 1, "C")

    for i, note in enumerate(notes_data):
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Section Start: ~{note['timestamp_str']}", 0, 1, "L")
        pdf.ln(5)
        pdf.image(note["image_path"], x=15, y=None, w=180)
        pdf.ln(5)
        if note["transcript"]:
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(0, 5, note["transcript"])

    pdf.output(output_pdf_path)
    print(f"Successfully created {output_pdf_path}")

def _group_transcript_by_keyframes(transcript: list, keyframes: list, pause_threshold: float) -> list:
    """Groups transcript snippets into paragraphs, forcing a break on each new keyframe."""
    if not transcript: return []
    
    paragraphs = []
    current_paragraph_text = ""
    current_paragraph_start = transcript[0]['start']
    keyframe_iter = iter(keyframes)
    next_keyframe = next(keyframe_iter, None)

    for i, snippet in enumerate(transcript):
        if next_keyframe and snippet['start'] >= next_keyframe['timestamp']:
            if current_paragraph_text.strip():
                paragraphs.append({"text": current_paragraph_text.strip(), "start_time": current_paragraph_start})
            current_paragraph_text = ""
            current_paragraph_start = next_keyframe['timestamp']
            next_keyframe = next(keyframe_iter, None)

        current_paragraph_text += snippet['text'] + " "
        
        is_last_snippet = (i == len(transcript) - 1)
        pause_duration = 0
        if not is_last_snippet:
            next_snippet = transcript[i+1]
            pause_duration = next_snippet['start'] - (snippet['start'] + snippet['duration'])

        if is_last_snippet or pause_duration >= pause_threshold:
            if current_paragraph_text.strip():
                paragraphs.append({"text": current_paragraph_text.strip(), "start_time": current_paragraph_start})
            if not is_last_snippet:
                current_paragraph_text = ""
                current_paragraph_start = next_snippet['start']
            
    return paragraphs

# --- Worker Process Functions ---

def _init_worker(model_id, device_str):
    """Initializer for each worker process."""
    global worker_model, worker_processor, worker_device
    print(f"Initializing worker {os.getpid()}...")
    worker_device = torch.device(device_str)
    worker_model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(worker_device)
    worker_processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True, use_fast=True)
    print(f"Worker {os.getpid()} initialized.")

def _process_chunk(chunk_id, video_stream_url, start_sec, end_sec, output_folder):
    """The main function executed by each worker process to find keyframes in a video segment."""
    print(f"Worker {os.getpid()} starting on chunk {chunk_id} ({_format_timestamp(start_sec)} to {_format_timestamp(end_sec)})")
    
    vid_capture = cv2.VideoCapture(video_stream_url)
    video_fps = vid_capture.get(cv2.CAP_PROP_FPS) or 30
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)
    vid_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_skip_interval = max(1, int(video_fps / FRAMES_TO_PROCESS_PER_SECOND))
    cooldown_frames = KEYFRAME_COOLDOWN_SECONDS * video_fps
    
    chunk_keyframes = []
    last_saved_embedding = None
    cooldown_counter = 0
    batch_data = []
    frame_index = start_frame

    while frame_index < end_frame:
        success, frame = vid_capture.read()
        if not success: break

        if cooldown_counter > 0:
            cooldown_counter -= 1
            frame_index += 1
            continue

        if frame_index % frame_skip_interval == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            batch_data.append({"pil_image": pil_image, "original_frame": frame, "frame_index": frame_index})

        if (len(batch_data) == BATCH_SIZE) or (not success and len(batch_data) > 0 and frame_index >= end_frame -1):
            if not batch_data: continue
            
            pil_images_for_batch = [item['pil_image'] for item in batch_data]
            with torch.no_grad():
                inputs = worker_processor(images=pil_images_for_batch, return_tensors="pt").to(worker_device)
                image_features = worker_model.get_image_features(**inputs)

            for i, embedding in enumerate(image_features):
                if cooldown_counter > 0: continue
                embedding = embedding.unsqueeze(0)
                is_keyframe = False
                
                if last_saved_embedding is None:
                    is_keyframe = True
                else:
                    similarity = cos_sim(embedding, last_saved_embedding)
                    if similarity.item() < SIMILARITY_THRESHOLD:
                        is_keyframe = True

                if is_keyframe:
                    current_item = batch_data[i]
                    current_keyframe_time = current_item['frame_index'] / video_fps
                    timestamp_str = _format_timestamp(current_keyframe_time)
                    safe_timestamp_str = timestamp_str.replace(":", "-")
                    frame_filename = os.path.join(output_folder, f"keyframe_{chunk_id}_{len(chunk_keyframes):04d}_at_{safe_timestamp_str}.png")
                    cv2.imwrite(frame_filename, current_item['original_frame'])
                    
                    print(f"Worker {os.getpid()} found keyframe at {timestamp_str}")
                    
                    chunk_keyframes.append({"image_path": frame_filename, "timestamp": current_keyframe_time})
                    last_saved_embedding = embedding
                    cooldown_counter = cooldown_frames
            
            batch_data = []
        frame_index += 1

    vid_capture.release()
    return chunk_keyframes

# --- Main Public Function ---

def generate_video_notes(video_url: str, output_dir: str = "."):
    """
    Generates a professional PDF with keyframes and transcript from a YouTube video.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Created temporary workspace: {temp_dir}")
        
        print("Pre-loading model in main process to cache it...")
        CLIPModel.from_pretrained(MODEL_ID)
        CLIPProcessor.from_pretrained(MODEL_ID, use_fast=True)
        print("Model cached successfully.")

        info_dict, stream_url = _get_video_stream_url_and_info(video_url)
        video_id = info_dict.get('id', None)
        video_duration = info_dict.get('duration', 0)

        raw_transcript = _get_transcript(video_id)

        all_keyframes = _identify_keyframes_parallel(stream_url, video_duration, temp_dir)
        
        final_notes_data = _associate_transcript_with_keyframes(raw_transcript, all_keyframes)

        video_title = info_dict.get('title', 'video_notes')
        sanitized_title = _sanitize_filename(video_title)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        final_filename = f"{sanitized_title}_{timestamp_str}.pdf"
        output_pdf_path = os.path.join(output_dir, final_filename)

        _create_pdf_report(final_notes_data, info_dict, video_url, output_pdf_path)

    finally:
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("Cleanup complete.")

# --- Main Execution Block ---

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    TEST_VIDEO_URL = 'https://www.youtube.com/watch?v=tURxphunUyk'
    OUTPUT_DIRECTORY = "./youtube_agent/generated_notes" 
    
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    generate_video_notes(video_url=TEST_VIDEO_URL, output_dir=OUTPUT_DIRECTORY)