import os
import json
import subprocess
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

def download_video(url, output_dir):
    """Download video and audio using yt-dlp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # We'll download the best video+audio and merge them into mp4
    # We'll also extract the audio as a separate wav file for AudioStack
    cmd = [
        "./venv/bin/yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", f"{output_dir}/%(title)s.%(ext)s",
        # "--write-auto-subs", # Let's get transcripts separately for better control
        url
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error downloading video: {result.stderr}")
        return None
    
    # Find the downloaded file
    for file in output_dir.glob("*.mp4"):
        if "The pain of becoming yourself" in file.name:
            return file
    return None

def get_transcript(video_id, output_path):
    """Fetch transcript using youtube-transcript-api."""
    try:
        print(f"Fetching transcript for video ID: {video_id}")
        # In this version, we MUST instantiate the class
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        transcript = transcript_list.find_transcript(['en']).fetch()
        
        formatter = JSONFormatter()
        json_formatted = formatter.format_transcript(transcript)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_formatted)
        print(f"Transcript saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return False

def main():
    url = "https://www.youtube.com/watch?v=QYAnJ_QyCQg"
    video_id = "QYAnJ_QyCQg"
    raw_dir = Path("data/raw/targeted_video")
    
    # 1. Download Video
    video_file = download_video(url, raw_dir)
    if not video_file:
        print("Failed to download video.")
        return

    # 2. Get Transcript
    transcript_file = raw_dir / "transcript.json"
    if not get_transcript(video_id, transcript_file):
        print("Failed to get transcript.")
        # Proceed anyway as we have the video

    print(f"\nTargeted acquisition complete!")
    print(f"Video: {video_file}")
    print(f"Transcript: {transcript_file}")

if __name__ == "__main__":
    main()
