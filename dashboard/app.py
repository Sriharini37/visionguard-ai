"""
VisionGuard AI — Streamlit Dashboard

Two modes:
  * DEMO MODE (default, always available): synthesizes realistic defect
    imagery + detections so the full UX can be shown live with zero setup.
  * REAL MODEL MODE: only offered when a live FastAPI backend with loaded
    weights is actually reachable. On the public cloud deployment (no API
    attached), this option never appears — so there is nothing to error on.

Run:
    streamlit run dashboard/app.py
"""
import io
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

st.set_page_config(page_title="VisionGuard AI", page_icon="🔍", layout="wide")

DEFECT_CLASSES = ["scratch", "dent", "crack", "discoloration", "solder_bridge", "missing_component"]
DEFAULT_API_URL = "http://localhost:8000"


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, class, confidence, verdict
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


# --------------------------------------------------------------------------
# Demo-mode synthetic data (no model weights required)
# --------------------------------------------------------------------------
def synthesize_product_image(has_defect: bool, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (416, 416), color=(230, 230, 235))
    draw = ImageDraw.Draw(img)

    # base "product" shape
    draw.rounded_rectangle([60, 60, 356, 356], radius=20, fill=(200, 205, 215), outline=(150, 150, 160), width=3)

    # texture noise
    noise = rng.normal(0, 8, (416, 416, 3))
    arr = np.array(img).astype(np.float32) + noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    box = None
    defect_type = None
    if has_defect:
        defect_type = rng.choice(DEFECT_CLASSES)
        x1, y1 = rng.integers(90, 250), rng.integers(90, 250)
        w, h = rng.integers(25, 70), rng.integers(25, 70)
        x2, y2 = x1 + w, y1 + h
        draw = ImageDraw.Draw(img)
        if defect_type in ("scratch", "crack"):
            draw.line([x1, y1, x2, y2], fill=(90, 30, 30), width=3)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=(120, 40, 40))
        box = (x1 - 10, y1 - 10, x2 + 10, y2 + 10)

    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img, box, defect_type


def synthesize_gradcam_overlay(img: Image.Image, box) -> Image.Image:
    """Fake but visually plausible heatmap centered on the defect box."""
    overlay = img.convert("RGBA")
    heat = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(heat)
    if box:
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        for r, alpha in zip(range(60, 10, -10), range(20, 160, 25)):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0, alpha))
    heat = heat.filter(ImageFilter.GaussianBlur(8))
    return Image.alpha_composite(overlay, heat).convert("RGB")


def run_demo_inference(seed: int, defect_rate: float):
    rng = np.random.default_rng(seed)
    has_defect = rng.random() < defect_rate
    img, box, defect_type = synthesize_product_image(has_defect, seed)
    confidence = float(rng.uniform(0.82, 0.995)) if has_defect else float(rng.uniform(0.9, 0.999))
    return img, box, defect_type, confidence


# --------------------------------------------------------------------------
# API availability probe — cached briefly so every rerun doesn't re-dial
# out and add latency, especially on the public cloud deployment where it
# will always fail.
# --------------------------------------------------------------------------
@st.cache_data(ttl=15, show_spinner=False)
def probe_api(url: str) -> bool:
    try:
        resp = requests.get(f"{url}/health", timeout=1.5)
        resp.raise_for_status()
        return bool(resp.json().get("model_loaded", False))
    except requests.exceptions.RequestException:
        return False


# --------------------------------------------------------------------------
# Sidebar — controls
# --------------------------------------------------------------------------
st.sidebar.title("🔍 VisionGuard AI")
st.sidebar.caption("Autonomous defect detection & explainable QA")

api_is_up = probe_api(st.session_state.api_url)

mode_options = ["Demo (synthetic)"]
if api_is_up:
    mode_options.append("Real model (live API)")

mode = st.sidebar.radio("Mode", mode_options, index=0)

domain = st.sidebar.selectbox("Production line", ["Electronics (PCB)", "Pharmaceutical tablets",
                                                    "Textile/apparel", "Solar panels"])
defect_rate = st.sidebar.slider("Simulated defect rate", 0.0, 1.0, 0.35, 0.05)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM overlay", value=True)
alert_threshold = st.sidebar.slider("Alert if defect rate/hour exceeds", 0.0, 1.0, 0.5, 0.05)

# uploaded_image is ALWAYS defined (None when not in Real mode), so
# downstream code never needs a fragile 'in dir()' / NameError-guard check.
uploaded_image = None
api_url = st.session_state.api_url

if mode.startswith("Real"):
    api_url = st.sidebar.text_input("API URL", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
        probe_api.clear()  # force a fresh check against the new URL
    uploaded_image = st.sidebar.file_uploader("Upload a product image to inspect",
                                               type=["jpg", "jpeg", "png"])
    st.sidebar.success("✅ Connected — real model loaded")
else:
    st.sidebar.caption(
        "Running in demo mode with synthetic data — no live model backend is "
        "attached here. To run real inference, start the API locally with "
        "`uvicorn api.main:app --port 8000` and reload this page."
    )

st.sidebar.divider()
run_button = st.sidebar.button("▶ Inspect next item", use_container_width=True)
run_batch = st.sidebar.button("⏩ Simulate 20-item batch", use_container_width=True)


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("VisionGuard AI — Live Quality Inspection")
st.caption(f"Production line: **{domain}**  |  Mode: **{mode}**")

col_live, col_stats = st.columns([2, 1])


def call_real_api(api_url: str, image_bytes: bytes):
    """Sends an image to the real FastAPI /inspect endpoint and normalizes
    the response into the same (img, box, defect_type, confidence, verdict)
    shape the demo path returns, so the rest of the UI code doesn't care
    which mode produced the result."""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    resp = requests.post(f"{api_url}/inspect", files=files, timeout=10)
    resp.raise_for_status()
    result = resp.json()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    detections = result.get("detections", [])
    box, defect_type, confidence = None, None, None
    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        box = tuple(top["bbox_xyxy"])
        defect_type = top["class"]
        confidence = top["confidence"]
    verdict = result.get("verdict", "PASS")
    return img, box, defect_type, confidence, verdict


def do_one_inspection():
    if mode.startswith("Real") and api_is_up and uploaded_image is not None:
        img, box, defect_type, confidence, verdict = call_real_api(api_url, uploaded_image.getvalue())
    else:
        seed = int(time.time() * 1000) % 100000
        img, box, defect_type, confidence = run_demo_inference(seed, defect_rate)
        verdict = "DEFECT" if box else "PASS"

    st.session_state.history.append({
        "timestamp": datetime.now(),
        "class": defect_type or "none",
        "confidence": confidence if confidence is not None else 0.0,
        "verdict": verdict,
    })
    return img, box, defect_type, confidence, verdict


if run_button:
    img, box, defect_type, confidence, verdict = do_one_inspection()

    with col_live:
        st.subheader("Latest inspection")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(img, caption="Camera feed (bounding box overlay)", use_container_width=True)
            if box:
                draw = ImageDraw.Draw(img)
                draw.rectangle(box, outline=(255, 0, 0), width=3)
                st.image(img, caption="Detection", use_container_width=True)
        with img_col2:
            if show_gradcam:
                overlay = synthesize_gradcam_overlay(img, box)
                st.image(overlay, caption="Grad-CAM explainability overlay", use_container_width=True)

        if verdict == "DEFECT":
            st.error(f"**DEFECT DETECTED** — {defect_type} (confidence: {confidence:.1%})")
        else:
            st.success(f"**PASS** (confidence: {confidence:.1%})")

elif run_batch:
    progress = st.progress(0, text="Running batch inspection...")
    for i in range(20):
        do_one_inspection()
        progress.progress((i + 1) / 20)
    progress.empty()
    st.toast("Batch of 20 items inspected", icon="✅")


# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
with col_stats:
    st.subheader("Session analytics")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        total = len(df)
        defects = (df["verdict"] == "DEFECT").sum()
        pass_rate = 1 - defects / total

        m1, m2, m3 = st.columns(3)
        m1.metric("Items inspected", total)
        m2.metric("Defects found", defects)
        m3.metric("Pass rate", f"{pass_rate:.1%}")

        st.bar_chart(df[df["verdict"] == "DEFECT"]["class"].value_counts())

        if defects / total > alert_threshold and total >= 5:
            st.warning(f"⚠️ Alert: defect rate ({defects/total:.1%}) exceeds threshold "
                       f"({alert_threshold:.0%}). Simulated Slack/email alert triggered.")
    else:
        st.info("Run an inspection to populate analytics.")

st.divider()

# --------------------------------------------------------------------------
# History table + PDF-style report export
# --------------------------------------------------------------------------
st.subheader("Inspection log")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download shift log (CSV)", csv, "visionguard_shift_log.csv", "text/csv")
else:
    st.caption("No inspections logged yet this session.")

st.divider()
with st.expander("ℹ️ About this dashboard"):
    st.markdown("""
    - **Demo mode** synthesizes plausible product images, defect boxes, and Grad-CAM
      heatmaps so the full workflow can be demonstrated without GPU hardware or trained weights.
    - **Real model mode** only appears when a live FastAPI backend with loaded
      weights is actually reachable — otherwise there's nothing to show, so it's
      hidden rather than erroring. Swap in `src/explainability/gradcam_utils.py`'s
      `get_gradcam_overlay()` for real heatmaps, and `ultralytics.YOLO(weights).predict()`
      for real detections.
    - Alerts here are simulated in-app; `docs/architecture.md` documents the real
      Slack/email webhook integration point.
    """)
