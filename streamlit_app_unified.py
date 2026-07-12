# ==========================================================
# STREAMLIT UNIFIED PIPELINE
# Binary (SimpleAdapter + LoRA) + Summary (Qwen3-VL + Q-Former LoRA)
# + Multi-class (EnhancedTemporalAdapterModel using generated summary)
# ==========================================================
import os
import tempfile

import streamlit as st

import pipeline as core
import binary_ensemble

st.set_page_config(
    page_title="Unified Video Analysis Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

VIDEO_DIR = os.getenv("VIDEO_DIR", os.path.join(os.getcwd(), "videos"))
SAVE_DIR = os.getenv("SAVE_DIR", os.path.join(os.getcwd(), "saved"))

# ==========================================================
# LOAD MODELS (CACHED)
# ==========================================================

@st.cache_resource
def load_all_models():
    try:
        with st.spinner("Loading models (this may take a few minutes)..."):
            return core.load_models(progress=None)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info(
            "Ensure binary weights, multi-class checkpoint and summarization checkpoint "
            "are reachable from the configured paths."
        )
        return None


# ==========================================================
# ANALYSIS
# ==========================================================

def analyze_video(video_path, models):
    """Run the unified pipeline on a single video."""
    with st.spinner("Analyzing video..."):
        return core.analyze_video(video_path, models)


# ==========================================================
# RESULTS + VQA RENDERER
# ==========================================================

def render_results(state_key, source_label):
    ctx = st.session_state[state_key]

    st.success("Analysis Complete")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Classification Results")
        st.metric("Status", ctx["binary_class"], f"{ctx['binary_confidence']:.2%}")
        st.caption(
            f"SimpleAdapter: {ctx['binary_parts']['simple']:.4f} | "
            f"LoRA: {ctx['binary_parts']['lora']:.4f}"
        )
        st.subheader("Attributes")
        st.markdown(f"- **Activity:** {ctx['activity']}")
        st.markdown(f"- **Category:** {ctx['category']}")
        st.markdown(f"- **Weapon:** {ctx['weapon']}")
        st.markdown(f"- **Location:** {ctx['location']}")
        st.markdown(f"- **People:** {ctx['people']}")
        st.markdown(f"- **Actions:** {', '.join(ctx['actions']) or 'None detected'}")

    with col2:
        st.subheader("Summary")
        st.write(ctx["summary"])
        if st.button("Save Summary", key=f"{state_key}_save"):
            os.makedirs(SAVE_DIR, exist_ok=True)
            base = os.path.splitext(os.path.basename(source_label))[0]
            txt_path = os.path.join(SAVE_DIR, f"{base}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(ctx["summary"])
            st.success(f"Saved to {txt_path}")

    # VQA
    st.divider()
    st.subheader("Ask Questions")
    col1, col2, col3, col4, col5 = st.columns(5)
    quick = None
    if col1.button("People", key=f"{state_key}_b_ppl"):
        quick = "How many people?"
    if col2.button("Weapon", key=f"{state_key}_b_wpn"):
        quick = "What weapon is used?"
    if col3.button("Location", key=f"{state_key}_b_loc"):
        quick = "Where is the event?"
    if col4.button("Category", key=f"{state_key}_b_cat"):
        quick = "What is the event category?"
    if col5.button("Actions", key=f"{state_key}_b_act"):
        quick = "What actions are detected?"

    question = st.text_input("Or type a question:", key=f"{state_key}_q")
    if quick is not None:
        question = quick

    if question:
        answer = core.answer_question(ctx, question)
        st.info(f"**Q:** {question}\n\n**A:** {answer}")


# ==========================================================
# STREAMLIT UI
# ==========================================================

st.title("Unified Video Analysis Pipeline")
st.markdown(
    """
Analyze surveillance videos with a single pipeline:
- Binary classification (normal/anomalous)
- AI-generated summary
- Multi-class attributes (activity, weapon, location, people, actions)
"""
)

st.sidebar.header("Configuration")
st.sidebar.info(
    f"""
**Qwen model:** {core.QWEN_MODEL_ID}
**Summarization checkpoint:** {core.SUMMARIZATION_CHECKPOINT}
**Multi-class dir:** {core.MULTICLASS_MODEL_DIR}
**Binary model dir:** {core.BINARY_MODEL_DIR}
**Device:** {core.DEVICE}
**Frames:** {core.FRAMES}
"""
)

tab1, tab2, tab3 = st.tabs(["Upload Video", "Local Video", "Settings"])

models = load_all_models()
if models is None:
    st.error("Failed to load models. Check your paths and dependencies.")
    st.stop()

# TAB 1: Upload Video
with tab1:
    st.header("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Analyze Video", key="analyze_upload"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            try:
                ctx = analyze_video(tmp_path, models)
            finally:
                os.unlink(tmp_path)

            if ctx is None:
                st.error("Could not read frames from the video.")
            else:
                ctx["source_label"] = uploaded_file.name
                st.session_state["ctx_upload"] = ctx

        if "ctx_upload" in st.session_state:
            render_results("ctx_upload", st.session_state["ctx_upload"].get("source_label", "video"))

# TAB 2: Local Video
with tab2:
    st.header("Select Local Video")

    if os.path.exists(VIDEO_DIR):
        video_files = []
        for root, _dirs, files in os.walk(VIDEO_DIR):
            for file in files:
                if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_files.append(os.path.join(root, file))

        if video_files:
            selected_video = st.selectbox("Choose a video:", video_files)

            if st.button("Analyze Video", key="analyze_local"):
                ctx = analyze_video(selected_video, models)
                if ctx is None:
                    st.error("Could not read frames from the video.")
                else:
                    ctx["source_label"] = selected_video
                    st.session_state["ctx_local"] = ctx

            if "ctx_local" in st.session_state:
                render_results("ctx_local", st.session_state["ctx_local"].get("source_label", "video"))
        else:
            st.warning("No video files found in the directory.")
    else:
        st.error(f"Video directory not found: {VIDEO_DIR}")

# TAB 3: Settings
with tab3:
    st.header("Settings")
    st.write("Current Configuration:")
    st.json({
        "QWEN_MODEL_ID": core.QWEN_MODEL_ID,
        "SUMMARIZATION_CHECKPOINT": core.SUMMARIZATION_CHECKPOINT,
        "MULTICLASS_MODEL_DIR": core.MULTICLASS_MODEL_DIR,
        "BINARY_MODEL_DIR": core.BINARY_MODEL_DIR,
        "BINARY_MODEL_DIR_MODULE": binary_ensemble.BINARY_MODEL_DIR,
        "DEVICE": core.DEVICE,
        "FRAMES": core.FRAMES,
    })
