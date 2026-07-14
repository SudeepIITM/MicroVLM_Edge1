🎬 MicroVLM Edge --- Video QA & Anomaly Detection System
A GPU-accelerated surveillance video analysis application built with
Streamlit and Qwen3-VL. The deployed hybrid pipeline performs binary
anomaly detection, structured multiclass analysis, video summarization,
and interactive video Q&A.
🌟 Features
🎥 Video Upload & Analysis --- Supports MP4, AVI, MOV, and MKV
video files
📊 Binary Anomaly Detection --- Predicts Normal vs Anomalous
with confidence
🔍 Hybrid Multiclass Classification --- Predicts activity,
weapon, people count, and location
🤖 Video Summarization --- Generates concise summaries using
Qwen3-VL-Instruct with LoRA adaptation
💬 Interactive Q&A --- Ask structured questions about the
analyzed surveillance video
💾 Export Results --- Save analysis results as a text report
⚡ GPU Accelerated --- PyTorch/CUDA-based inference
🎨 Streamlit Interface --- Interactive browser-based
demonstration UI
🏗️ Deployed Hybrid Architecture
``` text
Video Input
    ↓
8 Uniformly Sampled Frames
    ↓
    ├───────────────────────────────┐
    │                               │
    ↓                               ↓
Binary Anomaly Detection       Video Summarization
Normal / Anomalous             Qwen3-VL + LoRA
    │                               │
    └───────────────┬───────────────┘
                    ↓
          Hybrid Multiclass Analysis
                    │
        ┌───────────┴────────────┐
        │                        │
        ↓                        ↓
Activity + Weapon          People + Location
Previous trained           Temporal Adapter
classifiers                + Summary Fusion
        │                        │
        └───────────┬────────────┘
                    ↓
          Interactive Q&A & Results
```
Multiclass task mapping
Task       Deployed model path
---
Activity   Previous `clf_action_fine` classifier
Weapon     Previous `clf_weapon` classifier
People     Temporal Adapter + Summary Fusion
Location   Temporal Adapter + Summary Fusion
For Activity and Weapon, the previous classifier branch uses a fused
visual and summary representation. People and Location use the multitask
temporal-summary architecture.
📁 Project Structure
``` text
MicroVLM_Edge1/
├── app_core1.py
├── streamlit_app_v4.py
├── binary_ensemble.py
├── requirements.txt
├── final_hybrid_streamlit_deployment_v4.ipynb
└── README.md
```
Main files
`streamlit_app_v4.py` --- Streamlit UI and end-to-end pipeline
orchestration
`app_core1.py` --- Model loading, frame sampling, summarization,
feature extraction, and hybrid multiclass inference
`binary_ensemble.py` --- Binary Normal/Anomalous inference
`requirements.txt` --- Python dependencies
`final_hybrid_streamlit_deployment_v4.ipynb` --- Google Colab
deployment workflow
🚀 Recommended Deployment: Google Colab
The current deployment is designed to run from Google Colab with GPU
acceleration.
1. Clone the repository
``` bash
git clone https://github.com/SudeepIITM/MicroVLM_Edge1.git
cd MicroVLM_Edge1
```
2. Open the deployment notebook
Use:
``` text
final_hybrid_streamlit_deployment_v4.ipynb
```
The notebook:
Mounts Google Drive
Clones or updates this GitHub repository
Installs required dependencies
Copies trained model artifacts from Google Drive
Verifies all required model files
Configures the hybrid model paths
Starts `streamlit_app_v4.py`
Creates a public Cloudflare Quick Tunnel for the demo
3. Run all cells
Use a GPU runtime and run the notebook cells from top to bottom.
🔧 Model Artifact Configuration
Large trained model artifacts are intentionally not stored in
GitHub.
The deployment notebook expects model artifacts from Google Drive.
Binary model artifacts
``` text
/content/drive/MyDrive/models_ucf_v2
```
Multitask Temporal + Summary Fusion artifacts
``` text
/content/drive/MyDrive/Project_VLM/UCF_Multitask_Temporal
```
Expected files include:
``` text
temporal_adapter.pkl
summary_projector.pkl
fusion.pkl
activity_head.pkl
weapon_head.pkl
people_head.pkl
location_head.pkl
label_encoders.pkl
multitask_temporal_model.pkl
training_metadata.json
```
Previous Activity and Weapon classifier bundle
``` text
/content/drive/MyDrive/models3/multiclass_bundle.pkl
```
The bundle must contain:
``` text
clf_action_fine
le_action_fine
clf_weapon
le_weapon
```
Qwen3-VL + LoRA summarization checkpoint
The deployment notebook copies the trained summarization checkpoint from
the configured Google Drive path.
Expected checkpoint artifacts include:
``` text
adapter_config.json
adapter_model.safetensors
```
> Model paths can be changed in the configuration section of the
> deployment notebook.
📦 Main Dependencies
See `requirements.txt` for the complete environment.
Key technologies include:
Streamlit --- Interactive web application
PyTorch --- Deep-learning inference and temporal modules
Transformers --- Qwen3-VL model and processor loading
PEFT --- LoRA adapter loading
OpenCV --- Video decoding and frame sampling
Pillow --- Frame conversion to image objects
NumPy --- Numerical and frame-index operations
scikit-learn --- Saved classifier and label-processing support
joblib --- Loading trained classifier bundles
safetensors --- Model checkpoint support
🎥 Pipeline Execution
The main orchestration is implemented in `streamlit_app_v4.py`.
Basic analysis
``` python
def analyze_basic(video_path, models):
    frames = core.sample_frames(video_path)

    binary_class, binary_conf, binary_parts = (
        binary_ensemble.predict_binary(...)
    )

    summary = core.summarize_video(
        frames,
        models,
    )
```
This stage performs:
``` text
Video
  ↓
Frame Sampling
  ↓
Binary Anomaly Detection
  ↓
Video Summarization
```
Hybrid multiclass analysis
``` python
core.predict_multiclass(
    ctx["frames"],
    ctx["summary"],
    models,
)
```
The backend implementation is in `app_core1.py`.
``` text
Frames + Summary
       ↓
Hybrid Multiclass Pipeline
       ↓
Activity | Weapon | People | Location
```
🎞️ Frame Sampling
The deployed pipeline samples 8 approximately uniformly spaced
frames across the complete video.
``` python
indices = np.linspace(
    0,
    total_frames - 1,
    NUM_FRAMES,
    dtype=int,
)
```
This provides temporal coverage from the beginning to the end of the
video while reducing repeated full-video processing.
💻 Usage
Analyze a video
Open the Streamlit public URL generated by the deployment notebook
Select a sample video or upload a supported video
Click Analyze Video
Review the binary anomaly result and generated summary
Ask structured questions about activity, weapon, people, or location
Example questions
`What is the status?`
`Is the video anomalous?`
`What activity is happening?`
`Is there a weapon?`
`How many people are present?`
`Where is this happening?`
`Summarize the video.`
🖥️ Hardware and GPU Acceleration
The application uses PyTorch and CUDA for GPU-accelerated inference when
a compatible NVIDIA GPU is available.
GPU execution is used for:
Qwen3-VL visual-language inference
Frame-level feature extraction
LoRA-adapted summary generation
Temporal Adapter inference
Summary feature extraction and fusion
The deployment code also uses reduced-precision execution where
configured to improve GPU memory efficiency.
🐛 Troubleshooting
---
Issue                               Recommended action
---
`Missing model files`               Verify the Google Drive model paths
and required artifacts
`app_core1.py not found`            Confirm the latest repository
version has been cloned
`streamlit_app_v4.py not found`     Pull the latest `main` branch
CUDA out of memory                  Restart the runtime and close other
GPU workloads
Streamlit connection refused        Restart the Streamlit launch cell
before starting the tunnel
Cloudflare URL not generated        Stop the old tunnel and rerun the
Cloudflare tunnel cell
Cloudflare DNS error                Generate a fresh Quick Tunnel URL
GitHub 403 during push              Use a GitHub token with write
access to
`SudeepIITM/MicroVLM_Edge1`
🔐 Security and Repository Notes
Large model checkpoints are not committed to GitHub
Model artifacts are copied from configured Google Drive paths at
deployment time
GitHub access tokens should be stored in Colab Secrets and never
hard-coded
Uploaded videos are processed by the running application for
inference
Keep the Colab runtime active while using the public tunnel
📚 Code Demo Guide
For an end-to-end code demonstration, show the files in this order:
`streamlit_app_v4.py`
`analyze_basic()`
`core.sample_frames(...)`
`binary_ensemble.predict_binary(...)`
`core.summarize_video(...)`
`ensure_attrs()`
`core.predict_multiclass(...)`
`app_core1.py`
`sample_frames()`
`TemporalAdapter`
`SummaryProjector`
`FusionModule`
`predict_multiclass()`
Activity and Weapon classifier branch
People and Location temporal-summary fusion branch
📊 Current Deployment Summary
Component        Configuration
---
VLM Backbone     Qwen3-VL-2B-Instruct
Frame Sampling   8 uniformly sampled frames
Binary Output    Normal / Anomalous
Activity         Previous fine-grained activity classifier
Weapon           Previous weapon classifier
People           Temporal Adapter + Summary Fusion
Location         Temporal Adapter + Summary Fusion
Summarization    Qwen3-VL + LoRA
UI               Streamlit V4
Deployment       Google Colab + Cloudflare Quick Tunnel
📧 Contact
Author: Sudeep Rana  
GitHub: https://github.com/SudeepIITM  
Repository: https://github.com/SudeepIITM/MicroVLM_Edge1
🙏 Acknowledgments
Qwen Team --- Qwen3-VL models
Hugging Face --- Transformers and PEFT ecosystem
Streamlit --- Interactive web application framework
PyTorch --- Deep-learning framework
OpenCV --- Video processing
---
📌 Citation
If you use this project in research, please cite:
``` bibtex
@software{rana_microvlm_edge_2026,
  author = {Sudeep Rana},
  title = {MicroVLM Edge: Video QA and Anomaly Detection System},
  year = {2026},
  url = {https://github.com/SudeepIITM/MicroVLM_Edge1}
}
```
---
Built with Streamlit, PyTorch, and Qwen3-VL for edge-oriented
surveillance video understanding.
