import io
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Halaman Streamlit
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HitungBeras AI — YOLOv8 Pre-trained Rice Counter",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8_rice_pretrained.pt"
SAMPLES_DIR = BASE_DIR / "samples"

SAMPLE_FILES = {
    "Sampel 1 — Beras Sebar (~100 Butir)": SAMPLES_DIR / "sample_rice_real.png",
    "Sampel 2 — Kamera HP Atas (~48 Butir)": SAMPLES_DIR / "sample_rice_camera_1.jpg",
    "Sampel 3 — Kamera HP Kontras (~51 Butir)": SAMPLES_DIR / "sample_rice_camera_2.jpg",
    "Sampel 4 — Close-up Beras (~13 Butir)": SAMPLES_DIR / "sample_rice_closeup.jpg",
}

# Warna BGR untuk OpenCV
COLOR_UTUH = (129, 185, 16)       # Hijau Emerald (BGR)
COLOR_PATAH = (11, 158, 245)      # Kuning Amber (BGR)
COLOR_ASING = (68, 68, 239)       # Merah Rose (BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Pemuatan Model Pre-trained YOLOv8 (Cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model Pre-trained YOLOv8 khusus beras...")
def load_yolo_model(model_path: str) -> YOLO:
    return YOLO(model_path)


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi NMS (Non-Maximum Suppression) untuk Mode Tiling / Hybrid
# ─────────────────────────────────────────────────────────────────────────────
def non_max_suppression_boxes(
    detections: List[Dict], iou_thresh: float = 0.45
) -> List[Dict]:
    if not detections:
        return []

    boxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
    scores = np.array([d["conf"] for d in detections], dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        # Juga cek containment (jika box kecil berada di dalam box besar)
        min_area = np.minimum(areas[i], areas[order[1:]]) + 1e-6
        ioa = inter / min_area

        inds = np.where((iou <= iou_thresh) & (ioa <= 0.75))[0]
        order = order[inds + 1]

    return [detections[idx] for idx in keep]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Deteksi Standar dengan Pre-trained YOLOv8
# ─────────────────────────────────────────────────────────────────────────────
def detect_yolo_standard(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int = 640,
) -> List[Dict]:
    results = model.predict(
        source=img_bgr,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        verbose=False,
    )[0]

    detections: List[Dict] = []
    if results.boxes is None:
        return detections

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, cls_ids):
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        cls_name = results.names.get(cls_id, "Beras-Kepala")
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "area": float(w * h),
                "conf": float(conf),
                "cls_id": int(cls_id),
                "label": cls_name,
            }
        )

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# 2. Deteksi Sliced Tiling (SAHI-Style) dengan Pre-trained YOLOv8
#    Sangat akurat untuk foto dengan ratusan butir beras berukuran kecil
# ─────────────────────────────────────────────────────────────────────────────
def detect_yolo_sliced(
    model: YOLO,
    img_bgr: np.ndarray,
    conf_thresh: float,
    iou_thresh: float,
    grid_size: int = 2,
    overlap_ratio: float = 0.25,
) -> List[Dict]:
    h_img, w_img = img_bgr.shape[:2]

    # Jalankan deteksi full-frame terlebih dahulu
    all_dets = detect_yolo_standard(model, img_bgr, conf_thresh, iou_thresh, imgsz=640)

    # Bagi gambar ke dalam tile bertumpuk (overlapping tiles)
    step_x = w_img / grid_size
    step_y = h_img / grid_size
    pad_x = step_x * overlap_ratio
    pad_y = step_y * overlap_ratio

    for gy in range(grid_size):
        for gx in range(grid_size):
            tx1 = int(max(0, gx * step_x - pad_x))
            ty1 = int(max(0, gy * step_y - pad_y))
            tx2 = int(min(w_img, (gx + 1) * step_x + pad_x))
            ty2 = int(min(h_img, (gy + 1) * step_y + pad_y))

            tile = img_bgr[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            tile_dets = detect_yolo_standard(
                model, tile, conf_thresh, iou_thresh, imgsz=640
            )
            for d in tile_dets:
                bx1, by1, bx2, by2 = d["bbox"]
                gx1, gy1, gx2, gy2 = bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1

                # Abaikan box yang terpotong tepat di tepi dalam tile agar tidak duplikat
                margin = 3
                if (tx1 > 0 and bx1 < margin) or (ty1 > 0 and by1 < margin):
                    continue
                if (tx2 < w_img and bx2 > (tx2 - tx1 - margin)) or (
                    ty2 < h_img and by2 > (ty2 - ty1 - margin)
                ):
                    continue

                all_dets.append(
                    {
                        "bbox": [gx1, gy1, gx2, gy2],
                        "cx": (gx1 + gx2) / 2.0,
                        "cy": (gy1 + gy2) / 2.0,
                        "area": d["area"],
                        "conf": d["conf"],
                        "cls_id": d["cls_id"],
                        "label": d["label"],
                    }
                )

    return non_max_suppression_boxes(all_dets, iou_thresh=iou_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenCV Marker-Based Watershed Segmentation (Pendamping / Hybrid)
# ─────────────────────────────────────────────────────────────────────────────
def detect_opencv_watershed(
    img_bgr: np.ndarray, peak_ratio: float = 0.42
) -> List[Dict]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Jika mayoritas piksel putih (>65%), berarti background terang -> invert
    if np.mean(thresh > 0) > 0.65:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, sure_fg = cv2.threshold(dist, peak_ratio * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr.copy(), markers)

    h_img, w_img = img_bgr.shape[:2]
    min_grain_area = max(18, int((h_img * w_img) * 0.00008))
    max_grain_area = int((h_img * w_img) * 0.08)

    raw_blobs = []
    unique_labels = np.unique(markers)
    for lbl in unique_labels:
        if lbl <= 1:
            continue
        mask = np.uint8(markers == lbl) * 255
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < min_grain_area or area > max_grain_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        raw_blobs.append((x, y, x + w, y + h, area))

    if not raw_blobs:
        return []

    median_area = float(np.median([b[4] for b in raw_blobs]))
    detections: List[Dict] = []

    for x1, y1, x2, y2, area in raw_blobs:
        # Klasifikasi Beras-Kepala vs Beras-Patah berdasarkan rasio luas terhadap median
        is_whole = area >= (0.62 * median_area)
        detections.append(
            {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
                "area": float(area),
                "conf": 0.88,
                "cls_id": 1 if is_whole else 2,
                "label": "Beras-Kepala" if is_whole else "Beras-Patah",
            }
        )

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Visualisasi Annotasi pada Gambar
# ─────────────────────────────────────────────────────────────────────────────
def draw_annotations(
    img_bgr: np.ndarray,
    detections: List[Dict],
    viz_mode: str = "Bounding Box + Titik",
    include_foreign: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    annotated = img_bgr.copy()
    h, w = annotated.shape[:2]
    scale = max(0.45, min(w, h) / 900.0)
    thickness = max(1, int(round(scale * 2.2)))
    dot_radius = max(3, int(round(scale * 5.5)))

    counts = {
        "total_beras": 0,
        "beras_utuh": 0,
        "beras_patah": 0,
        "benda_asing": 0,
    }

    grain_idx = 0
    for d in detections:
        label = d["label"]
        x1, y1, x2, y2 = map(int, d["bbox"])
        cx, cy = int(round(d["cx"])), int(round(d["cy"]))

        if label == "Benda-Asing":
            counts["benda_asing"] += 1
            if not include_foreign:
                continue
            color = COLOR_ASING
        elif label == "Beras-Patah":
            counts["beras_patah"] += 1
            counts["total_beras"] += 1
            color = COLOR_PATAH
        else:
            counts["beras_utuh"] += 1
            counts["total_beras"] += 1
            color = COLOR_UTUH

        grain_idx += 1

        if viz_mode == "Bounding Box + Titik":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(annotated, (cx, cy), dot_radius, color, -1)
            cv2.circle(annotated, (cx, cy), dot_radius, (255, 255, 255), 1)
        elif viz_mode == "Titik Tengah (Dot Marker)":
            cv2.circle(annotated, (cx, cy), dot_radius + 2, (255, 255, 255), -1)
            cv2.circle(annotated, (cx, cy), dot_radius, color, -1)
        elif viz_mode == "Nomor Urut Butir":
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.circle(annotated, (cx, cy), dot_radius + 5, (15, 23, 42), -1)
            cv2.putText(
                annotated,
                str(grain_idx),
                (cx - dot_radius, cy + int(dot_radius * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.35, scale * 0.55),
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return annotated, counts


# ─────────────────────────────────────────────────────────────────────────────
# UI Utama Aplikasi Streamlit
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        """
        <style>
        .main-counter-box {
            background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(15,23,42,0.95));
            border: 1px solid rgba(16,185,129,0.45);
            border-radius: 16px;
            padding: 20px 24px;
            text-align: center;
            margin-bottom: 16px;
        }
        .main-counter-number {
            font-size: 3.4rem;
            font-weight: 800;
            color: #10b981;
            line-height: 1.1;
        }
        .main-counter-label {
            font-size: 0.95rem;
            color: #cbd5e1;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("🌾 HitungBeras AI — Penghitung Butir Beras (Pre-trained YOLOv8)")
    st.caption(
        "Menggunakan model **Pre-trained YOLOv8 Object Detection** (`yolov8_rice_pretrained.pt`) "
        "yang telah dilatih pada dataset butir beras untuk mendeteksi dan menghitung jumlah beras secara akurat."
    )

    # Load model
    if not MODEL_PATH.exists():
        st.error(f"File model tidak ditemukan di: `{MODEL_PATH}`")
        st.stop()

    model = load_yolo_model(str(MODEL_PATH))

    # ─── Sidebar Pengaturan ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Pengaturan Model & Deteksi")

        detection_mode = st.selectbox(
            "Metode Deteksi AI",
            options=[
                "Pre-trained YOLOv8 (Standar)",
                "Pre-trained YOLOv8 + Sliced Tiling (Foto Padat)",
                "Hybrid: YOLOv8 + OpenCV Watershed",
            ],
            index=0,
            help=(
                "- **YOLOv8 Standar**: Cepat & akurat untuk foto jarak dekat/menengah.\n"
                "- **YOLOv8 + Sliced Tiling**: Membagi gambar ke beberapa kotak kecil agar beras berukuran kecil/ratusan butir terdeteksi semua.\n"
                "- **Hybrid YOLOv8 + Watershed**: Menggabungkan deteksi YOLOv8 dan segmentasi kontur OpenCV Watershed."
            ),
        )

        conf_thresh = st.slider(
            "Confidence Threshold (Ambang Keyakinan)",
            min_value=0.10,
            max_value=0.85,
            value=0.25,
            step=0.05,
            help="Turunkan (misal 0.15 - 0.25) agar semua butir beras terdeteksi; naikkan jika ada objek latar yang salah terhitung.",
        )

        iou_thresh = st.slider(
            "IoU Threshold (Filter Tumpukan Box)",
            min_value=0.15,
            max_value=0.85,
            value=0.45,
            step=0.05,
            help="Mengatur batas Non-Maximum Suppression (NMS) antar kotak deteksi yang berdekatan.",
        )

        viz_mode = st.radio(
            "Tampilan Penanda Visual",
            options=[
                "Bounding Box + Titik",
                "Titik Tengah (Dot Marker)",
                "Nomor Urut Butir",
            ],
            index=0,
        )

        with st.expander("✂️ Potong Tepi Gambar (Crop ROI)", expanded=False):
            crop_margin = st.slider(
                "Potong Tepi Luar (%)",
                min_value=0,
                max_value=25,
                value=0,
                step=1,
                help="Gunakan jika pinggiran meja/wadah di tepi foto ikut terdeteksi.",
            )

        include_foreign = st.checkbox(
            "Tampilkan penanda Benda Asing (Batu/Gabah)", value=False
        )

    # ─── Tab Pilihan Sumber Gambar ────────────────────────────────────────────
    tab_sample, tab_upload, tab_camera = st.tabs(
        ["🖼️ Coba Sampel Foto Beras Asli", "📤 Unggah Foto Beras", "📷 Ambil Foto Kamera"]
    )

    selected_pil_image: Image.Image | None = None
    source_name = "sampel_beras.jpg"

    with tab_sample:
        col_s1, col_s2 = st.columns([2, 3])
        with col_s1:
            chosen_sample_label = st.selectbox(
                "Pilih Foto Sampel Beras:",
                options=list(SAMPLE_FILES.keys()),
                index=0,
            )
            sample_path = SAMPLE_FILES[chosen_sample_label]
            if sample_path.exists():
                selected_pil_image = Image.open(sample_path).convert("RGB")
                source_name = sample_path.name
        with col_s2:
            st.info(
                "💡 **Tips:** Anda dapat langsung menguji akurasi model **Pre-trained YOLOv8** "
                "pada 4 foto beras asli di samping, atau beralih ke tab **Unggah Foto Beras** / **Ambil Foto Kamera**."
            )

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Unggah foto butir beras (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_file is not None:
            selected_pil_image = Image.open(uploaded_file).convert("RGB")
            source_name = uploaded_file.name

    with tab_camera:
        camera_photo = st.camera_input("Ambil foto butir beras dengan kamera")
        if camera_photo is not None:
            selected_pil_image = Image.open(camera_photo).convert("RGB")
            source_name = "kamera_beras.jpg"

    if selected_pil_image is None:
        st.warning("Silakan pilih sampel atau unggah foto beras terlebih dahulu.")
        return

    # ─── Pra-pemrosesan & Crop ROI ────────────────────────────────────────────
    img_rgb = np.array(selected_pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if crop_margin > 0:
        h_raw, w_raw = img_bgr.shape[:2]
        dx = int(w_raw * (crop_margin / 100.0))
        dy = int(h_raw * (crop_margin / 100.0))
        img_bgr = img_bgr[dy : h_raw - dy, dx : w_raw - dx]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ─── Jalankan Inferensi Model ─────────────────────────────────────────────
    t_start = time.perf_counter()

    if detection_mode == "Pre-trained YOLOv8 (Standar)":
        detections = detect_yolo_standard(model, img_bgr, conf_thresh, iou_thresh)
    elif detection_mode == "Pre-trained YOLOv8 + Sliced Tiling (Foto Padat)":
        detections = detect_yolo_sliced(
            model, img_bgr, conf_thresh, iou_thresh, grid_size=2
        )
    else:
        yolo_dets = detect_yolo_standard(model, img_bgr, conf_thresh, iou_thresh)
        ws_dets = detect_opencv_watershed(img_bgr)
        detections = non_max_suppression_boxes(
            yolo_dets + ws_dets, iou_thresh=iou_thresh
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # Gambar hasil deteksi
    annotated_bgr, counts = draw_annotations(
        img_bgr, detections, viz_mode=viz_mode, include_foreign=include_foreign
    )
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # ─── Tampilan Hasil & Statistik ───────────────────────────────────────────
    col_img_orig, col_img_res = st.columns(2)

    with col_img_orig:
        st.subheader("📷 Foto Input")
        st.image(img_rgb, use_container_width=True, caption=f"Sumber: {source_name}")

    with col_img_res:
        st.subheader("🎯 Hasil Deteksi Model Pre-trained")
        st.image(
            annotated_rgb,
            use_container_width=True,
            caption=f"{detection_mode} • Waktu inferensi: {elapsed_ms:.0f} ms",
        )

    # Koreksi Manual Opsional & Kartu Total
    col_main_count, col_metrics = st.columns([1.2, 2.8])

    with col_main_count:
        manual_adj = st.number_input(
            "Koreksi Manual (+/- butir jika diperlukan):",
            min_value=-500,
            max_value=500,
            value=0,
            step=1,
        )
        final_total = max(0, counts["total_beras"] + int(manual_adj))

        st.markdown(
            f"""
            <div class="main-counter-box">
                <div class="main-counter-label">Total Jumlah Butir Beras</div>
                <div class="main-counter-number">{final_total}</div>
                <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                    Deteksi AI: {counts["total_beras"]} butir {f"({manual_adj:+d} manual)" if manual_adj != 0 else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_metrics:
        pct_utuh = (
            (counts["beras_utuh"] / counts["total_beras"] * 100.0)
            if counts["total_beras"] > 0
            else 0.0
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🟢 Beras Utuh (Kepala)", f"{counts['beras_utuh']} butir")
        m2.metric("🟡 Beras Patah", f"{counts['beras_patah']} butir")
        m3.metric("📊 Persentase Utuh", f"{pct_utuh:.1f}%")
        m4.metric("⚡ Waktu Proses", f"{elapsed_ms:.0f} ms")

        # Tombol Unduh Gambar Hasil & CSV
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="📥 Unduh Gambar Hasil Deteksi (PNG)",
                data=buf.getvalue(),
                file_name=f"hasil_hitung_beras_{final_total}_butir.png",
                mime="image/png",
                use_container_width=True,
            )

        if detections:
            df = pd.DataFrame(
                [
                    {
                        "No": i + 1,
                        "Kategori": d["label"],
                        "Confidence": round(d["conf"], 3),
                        "Posisi X": round(d["cx"], 1),
                        "Posisi Y": round(d["cy"], 1),
                        "Luas Box (px²)": round(d["area"], 1),
                    }
                    for i, d in enumerate(detections)
                ]
            )
            with dl_col2:
                st.download_button(
                    label="📄 Unduh Tabel Deteksi (CSV)",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"data_butir_beras_{final_total}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with st.expander("📋 Lihat Tabel Detail Setiap Butir Beras Terdeteksi"):
                st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
