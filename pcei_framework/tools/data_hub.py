"""
PCEI Data Hub — Unified Import + Label + Train Pipeline

A single Streamlit interface where you:
1. Upload raw files (mp4, png, jpg, mp3, wav, txt, pdf, md)
2. See all unlabeled data in a queue
3. Label each item with free-text descriptions and category tags
4. Send labeled items to the training database (JSONL)

Launch: streamlit run tools/data_hub.py
"""
import streamlit as st
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# ——— CONFIG ———
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_DB = PROJECT_ROOT / "data" / "processed" / "training_database.jsonl"
LABEL_CACHE = PROJECT_ROOT / "data" / "processed" / "label_cache.json"

# Auto-create directories
for sub in ["images", "audio", "video", "text"]:
    (RAW_DIR / sub).mkdir(parents=True, exist_ok=True)
TRAIN_DB.parent.mkdir(parents=True, exist_ok=True)

# File type mappings
EXT_MAP = {
    '.png': ('images', 'image'), '.jpg': ('images', 'image'), '.jpeg': ('images', 'image'),
    '.webp': ('images', 'image'),
    '.mp3': ('audio', 'audio'), '.wav': ('audio', 'audio'), '.ogg': ('audio', 'audio'),
    '.flac': ('audio', 'audio'),
    '.mp4': ('video', 'video'), '.mov': ('video', 'video'), '.avi': ('video', 'video'),
    '.webm': ('video', 'video'),
    '.txt': ('text', 'text'), '.pdf': ('text', 'text'), '.md': ('text', 'text'),
}

CATEGORY_TAGS = [
    "concept", "emotion", "object", "action", "scene",
    "music", "speech", "nature", "abstract", "reasoning",
    "conversation", "instruction", "narrative", "technical"
]

# ——— STATE ———
st.set_page_config(page_title="PCEI Data Hub", layout="wide", page_icon="🧠")

if 'label_cache' not in st.session_state:
    # Load existing labels
    if LABEL_CACHE.exists():
        with open(LABEL_CACHE, 'r') as f:
            st.session_state.label_cache = json.load(f)
    else:
        st.session_state.label_cache = {}

if 'queue_idx' not in st.session_state:
    st.session_state.queue_idx = 0


# ——— HELPERS ———
def get_all_raw_files():
    """Scan all raw data subdirectories for supported files."""
    files = []
    for ext, (subdir, modality) in EXT_MAP.items():
        for f in (RAW_DIR / subdir).rglob(f"*{ext}"):
            if f.is_file():
                files.append((f, modality))
    return sorted(files, key=lambda x: x[0].name)

def get_unlabeled_files():
    """Return only files that haven't been labeled yet."""
    all_files = get_all_raw_files()
    return [(f, m) for f, m in all_files if str(f) not in st.session_state.label_cache]

def get_labeled_files():
    """Return files that have been labeled but not yet sent to training DB."""
    all_files = get_all_raw_files()
    # Check which are in label_cache but not yet in training DB
    sent_files = set()
    if TRAIN_DB.exists():
        with open(TRAIN_DB, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        sent_files.add(record.get('filepath', ''))
                    except json.JSONDecodeError:
                        pass
    
    return [(f, m) for f, m in all_files 
            if str(f) in st.session_state.label_cache and str(f) not in sent_files]

def save_label(filepath: str, label: str, tags: list, modality: str):
    """Save label to cache (not yet to training DB)."""
    st.session_state.label_cache[filepath] = {
        "label": label,
        "tags": tags,
        "modality": modality,
        "labeled_at": datetime.utcnow().isoformat()
    }
    with open(LABEL_CACHE, 'w') as f:
        json.dump(st.session_state.label_cache, f, indent=2)

def send_to_training(filepath: str, label_data: dict):
    """Append a labeled item to the training database JSONL."""
    record = {
        "filepath": filepath,
        "label": label_data["label"],
        "tags": label_data["tags"],
        "modality": label_data["modality"],
        "labeled_at": label_data["labeled_at"],
        "sent_to_training": datetime.utcnow().isoformat()
    }
    with open(TRAIN_DB, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')

def read_text_file(filepath: Path) -> str:
    """Read text from txt, md, or pdf files."""
    ext = filepath.suffix.lower()
    if ext in ['.txt', '.md']:
        return filepath.read_text(encoding='utf-8', errors='replace')
    elif ext == '.pdf':
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(filepath))
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except ImportError:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(str(filepath))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return f"[PDF reading requires PyMuPDF or PyPDF2. Install: pip install PyMuPDF]"
    return ""


# ——— UI ———
st.title("🧠 PCEI Data Hub")
st.caption("Import → Label → Train. All in one place.")

tab_upload, tab_label, tab_review, tab_stats = st.tabs([
    "📤 Upload", "🏷️ Label Queue", "✅ Review & Send", "📊 Stats"
])

# ═══════════════════════════════════════════
# TAB 1: UPLOAD
# ═══════════════════════════════════════════
with tab_upload:
    st.subheader("Upload Raw Data")
    st.markdown("Drag files here. They'll be sorted into `data/raw/{images,audio,video,text}/` automatically.")
    
    uploaded = st.file_uploader(
        "Choose files",
        type=["png", "jpg", "jpeg", "webp", "mp3", "wav", "ogg", "flac",
              "mp4", "mov", "avi", "webm", "txt", "pdf", "md"],
        accept_multiple_files=True
    )
    
    if uploaded:
        for file in uploaded:
            ext = Path(file.name).suffix.lower()
            if ext in EXT_MAP:
                subdir, modality = EXT_MAP[ext]
                dest = RAW_DIR / subdir / file.name
                
                if not dest.exists():
                    with open(dest, 'wb') as f:
                        f.write(file.getbuffer())
                    st.success(f"✅ Saved `{file.name}` → `data/raw/{subdir}/`")
                else:
                    st.info(f"⏭️ `{file.name}` already exists, skipping.")
            else:
                st.warning(f"⚠️ Unsupported format: `{ext}`")

# ═══════════════════════════════════════════
# TAB 2: LABEL QUEUE
# ═══════════════════════════════════════════
with tab_label:
    st.subheader("Label Queue")
    
    unlabeled = get_unlabeled_files()
    
    if not unlabeled:
        st.success("🎉 All files are labeled! Head to the Review tab to send them to training.")
    else:
        st.info(f"**{len(unlabeled)}** files waiting for labels.")
        
        # Clamp index
        idx = min(st.session_state.queue_idx, len(unlabeled) - 1)
        filepath, modality = unlabeled[idx]
        
        # Navigation
        nav1, nav2, nav3 = st.columns([1, 3, 1])
        with nav1:
            if st.button("⬅️ Prev", use_container_width=True, disabled=idx == 0):
                st.session_state.queue_idx = max(0, idx - 1)
                st.rerun()
        with nav2:
            st.markdown(f"<h4 style='text-align:center'>{idx + 1} / {len(unlabeled)} — {modality.upper()}</h4>", 
                       unsafe_allow_html=True)
            st.caption(f"`{filepath.name}`")
        with nav3:
            if st.button("Next ➡️", use_container_width=True, disabled=idx >= len(unlabeled) - 1):
                st.session_state.queue_idx = min(len(unlabeled) - 1, idx + 1)
                st.rerun()
        
        st.divider()
        
        # Preview + Label side by side
        preview_col, label_col = st.columns([2, 1])
        
        with preview_col:
            ext = filepath.suffix.lower()
            if modality == "image":
                st.image(str(filepath), use_container_width=True)
            elif modality == "audio":
                st.audio(str(filepath))
            elif modality == "video":
                st.video(str(filepath))
            elif modality == "text":
                content = read_text_file(filepath)
                st.text_area("File Contents", content, height=400, disabled=True)
                
        with label_col:
            st.subheader("Annotation")
            
            label_text = st.text_area(
                "What is this? Describe it:",
                height=200,
                placeholder="A red sports car driving on a sunny highway..."
            )
            
            selected_tags = st.multiselect(
                "Category Tags:",
                CATEGORY_TAGS,
                default=[]
            )
            
            if st.button("💾 Save Label & Next", type="primary", use_container_width=True):
                if label_text.strip():
                    save_label(str(filepath), label_text.strip(), selected_tags, modality)
                    st.toast("Label saved!", icon="✅")
                    # Move to next
                    st.session_state.queue_idx = idx  # stays same idx since this file leaves the unlabeled list
                    st.rerun()
                else:
                    st.error("Label cannot be empty.")

# ═══════════════════════════════════════════
# TAB 3: REVIEW & SEND TO TRAINING
# ═══════════════════════════════════════════
with tab_review:
    st.subheader("Labeled Items Ready for Training")
    
    ready = get_labeled_files()
    
    if not ready:
        st.info("No labeled items pending. Upload and label data first.")
    else:
        st.success(f"**{len(ready)}** items ready to send to training database.")
        
        # Show a preview table
        for filepath, modality in ready:
            label_data = st.session_state.label_cache.get(str(filepath), {})
            with st.expander(f"📄 {filepath.name} ({modality})"):
                st.write(f"**Label:** {label_data.get('label', 'N/A')}")
                st.write(f"**Tags:** {', '.join(label_data.get('tags', []))}")
                st.write(f"**Labeled at:** {label_data.get('labeled_at', 'N/A')}")
        
        if st.button("🚀 Send ALL to Training Database", type="primary", use_container_width=True):
            count = 0
            for filepath, modality in ready:
                label_data = st.session_state.label_cache.get(str(filepath))
                if label_data:
                    send_to_training(str(filepath), label_data)
                    count += 1
            st.success(f"Sent {count} items to `{TRAIN_DB.name}`!")
            st.balloons()

# ═══════════════════════════════════════════
# TAB 4: STATS
# ═══════════════════════════════════════════
with tab_stats:
    st.subheader("Pipeline Statistics")
    
    all_files = get_all_raw_files()
    unlabeled = get_unlabeled_files()
    labeled_pending = get_labeled_files()
    
    # Count items in training DB
    train_count = 0
    if TRAIN_DB.exists():
        with open(TRAIN_DB, 'r') as f:
            train_count = sum(1 for line in f if line.strip())
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Raw Files", len(all_files))
    c2.metric("Unlabeled", len(unlabeled))
    c3.metric("Labeled (pending)", len(labeled_pending))
    c4.metric("In Training DB", train_count)
    
    # Modality breakdown
    st.divider()
    modality_counts = {}
    for _, m in all_files:
        modality_counts[m] = modality_counts.get(m, 0) + 1
    
    if modality_counts:
        st.bar_chart(modality_counts)
