import streamlit as st
import os
import json
from pathlib import Path
from datetime import datetime

# --- CONFIG ---
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DEFAULT_OUTPUT_FILE = Path(__file__).parent.parent / "data" / "processed" / "multimodal_labels.jsonl"

st.set_page_config(page_title="PCEI Multimodal Labeler", layout="wide")

# Ensure directories exist
DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Supported Extensions
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
AUDIO_EXTS = {'.wav', '.mp3', '.ogg', '.flac'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.webm'}
ALL_EXTS = IMAGE_EXTS | AUDIO_EXTS | VIDEO_EXTS


# --- STATE MANAGEMENT ---
if 'file_list' not in st.session_state:
    st.session_state.file_list = []
if 'current_idx' not in st.session_state:
    st.session_state.current_idx = 0
if 'labels' not in st.session_state:
    st.session_state.labels = {}


# --- HELPER FUNCTIONS ---
def get_media_files(directory: Path):
    files = []
    for ext in ALL_EXTS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted([f for f in files if f.is_file()])

def load_existing_labels(file_path: Path):
    labels = {}
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        labels[record['filepath']] = record['label']
                    except json.JSONDecodeError:
                        pass
    return labels

def save_label(filepath: str, label_text: str):
    record = {
        "filepath": str(filepath),
        "label": label_text,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save to session
    st.session_state.labels[str(filepath)] = label_text
    
    # Append to file
    with open(DEFAULT_OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

def next_file():
    if st.session_state.current_idx < len(st.session_state.file_list) - 1:
        st.session_state.current_idx += 1

def prev_file():
    if st.session_state.current_idx > 0:
        st.session_state.current_idx -= 1


# --- UI ---
st.title("PCEI Multimodal Labeler 🏷️")
st.markdown("Rapidly curate short clips, images, and audio waveforms for the `ContinuousWorldStack` and sensory stacks.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    data_dir_input = st.text_input("Raw Data Directory", str(DEFAULT_DATA_DIR))
    
    if st.button("Scan Directory"):
        path = Path(data_dir_input)
        if path.exists() and path.is_dir():
            files = get_media_files(path)
            st.session_state.file_list = files
            st.session_state.current_idx = 0
            st.session_state.labels = load_existing_labels(DEFAULT_OUTPUT_FILE)
            st.success(f"Found {len(files)} media files.")
        else:
            st.error("Directory not found.")

    st.markdown("---")
    st.metric("Total Files", len(st.session_state.file_list))
    st.metric("Labeled Files", len(st.session_state.labels))


# Main Labeling Area
if not st.session_state.file_list:
    st.info("👈 Please enter a valid directory containing media files and click 'Scan Directory'.")
else:
    # Navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.button("⬅️ Previous", on_click=prev_file, use_container_width=True)
    with col2:
        file_path = st.session_state.file_list[st.session_state.current_idx]
        is_labeled = str(file_path) in st.session_state.labels
        status = "✅ Labeled" if is_labeled else "⏳ Unlabeled"
        st.markdown(f"<h3 style='text-align: center;'>{status} ({st.session_state.current_idx + 1} / {len(st.session_state.file_list)})</h3>", unsafe_allow_html=True)
        st.caption(f"File: `{file_path.name}`")
    with col3:
        st.button("Next ➡️", on_click=next_file, use_container_width=True)

    # Media Display
    st.markdown("---")
    display_col, input_col = st.columns([2, 1])
    
    with display_col:
        ext = file_path.suffix.lower()
        if ext in IMAGE_EXTS:
            st.image(str(file_path), use_container_width=True)
        elif ext in AUDIO_EXTS:
            st.audio(str(file_path))
            st.info("Audio waveforms will be converted to Mel-Spectrograms during actual training ingest.")
        elif ext in VIDEO_EXTS:
            st.video(str(file_path))
            st.warning("Ensure videos are short (2-3 seconds max) to prevent M4 Air OOM errors during synchronous `ImageStack`/`AudioStack` diffusion.")

    with input_col:
        st.subheader("Annotation")
        current_label = st.session_state.labels.get(str(file_path), "")
        
        label_input = st.text_area(
            "What conceptually describes this media?", 
            value=current_label, 
            height=200,
            placeholder="A red car driving on a sunny road..."
        )
        
        if st.button("Save & Next 💾", type="primary", use_container_width=True):
            if label_input.strip():
                save_label(file_path, label_input.strip())
                st.toast("Label saved!", icon="✅")
                next_file()
                st.rerun()
            else:
                st.error("Label cannot be empty.")
