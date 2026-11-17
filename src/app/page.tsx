"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Camera,
  Upload,
  Sparkles,
  Sliders,
  Crop,
  Download,
  History,
  Trash2,
  RotateCcw,
  CheckCircle2,
  MousePointerClick,
  Eye,
  Save,
  X,
  Info,
} from "lucide-react";
import {
  detectRiceGrains,
  DetectedGrain,
  CropRect,
} from "@/lib/riceDetector";
import { generateSampleRicePhoto } from "@/lib/sampleGenerator";

interface HistoryItem {
  id: string;
  label: string;
  timestamp: string;
  totalCount: number;
  autoCount: number;
  manualAdded: number;
  manualRemoved: number;
  thumbnailUrl: string;
}

const STORAGE_KEY = "hitungberas:history:v1";

export default function HitungBerasPage() {
  // Image source state
  const [imageSrc, setImageSrc] = useState<string>("");
  const [sampleGroundTruth, setSampleGroundTruth] = useState<number | null>(null);
  const loadedImageRef = useRef<HTMLImageElement | null>(null);

  // Detection parameters
  const [threshold, setThreshold] = useState<number>(135);
  const [separationSensitivity, setSeparationSensitivity] = useState<number>(60);
  const [minGrainSize, setMinGrainSize] = useState<number>(16);
  const [crop, setCrop] = useState<CropRect>({
    x: 0.05,
    y: 0.05,
    width: 0.9,
    height: 0.9,
  });

  // Detection results & manual edits
  const [autoGrains, setAutoGrains] = useState<DetectedGrain[]>([]);
  const [removedAutoIds, setRemovedAutoIds] = useState<Set<string>>(new Set());
  const [manualGrains, setManualGrains] = useState<DetectedGrain[]>([]);
  const [maskDataUrl, setMaskDataUrl] = useState<string>("");
  const [processTimeMs, setProcessTimeMs] = useState<number>(0);

  // View & UI Controls
  const [viewMode, setViewMode] = useState<"dots" | "numbers" | "mask" | "original">("dots");
  const [activeTab, setActiveTab] = useState<"sensitivity" | "crop">("sensitivity");
  const [noteLabel, setNoteLabel] = useState<string>("");
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);

  // Refs
  const displayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const cameraDialogRef = useRef<HTMLDialogElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  // Load saved history from localStorage once on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      }
    } catch {
      // Ignore storage errors in restricted contexts
    }
  }, []);

  // Run detection pipeline on current image
  const runDetection = useCallback(
    (
      img: HTMLImageElement,
      nextParams: {
        threshold: number;
        separationSensitivity: number;
        minGrainSize: number;
        crop: CropRect;
      },
      autoOtsu = false
    ) => {
      const res = detectRiceGrains(img, nextParams, autoOtsu);
      setAutoGrains(res.grains);
      setMaskDataUrl(res.maskDataUrl);
      setProcessTimeMs(res.processTimeMs);
      if (autoOtsu) {
        setThreshold(res.computedOtsu);
      }
    },
    []
  );

  // Load a sample image on initial page load so user sees immediate live demo
  useEffect(() => {
    const sample = generateSampleRicePhoto(65);
    if (sample.dataUrl) {
      setSampleGroundTruth(sample.actualCount);
      setImageSrc(sample.dataUrl);
    }
  }, []);

  // Trigger image load & initial auto-Otsu detection whenever imageSrc changes
  useEffect(() => {
    if (!imageSrc) return;
    const img = new Image();
    img.onload = () => {
      loadedImageRef.current = img;
      setRemovedAutoIds(new Set());
      setManualGrains([]);
      runDetection(
        img,
        {
          threshold,
          separationSensitivity,
          minGrainSize,
          crop,
        },
        true
      );
    };
    img.src = imageSrc;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSrc]);

  // Re-run detection when parameters or crop change (preserving manual additions)
  useEffect(() => {
    const img = loadedImageRef.current;
    if (!img) return;
    runDetection(
      img,
      {
        threshold,
        separationSensitivity,
        minGrainSize,
        crop,
      },
      false
    );
  }, [threshold, separationSensitivity, minGrainSize, crop, runDetection]);

  // Active grains after applying Tap-to-Edit removals & additions
  const visibleAutoGrains = autoGrains.filter((g) => !removedAutoIds.has(g.id));
  const allActiveGrains = [...visibleAutoGrains, ...manualGrains];
  const totalCount = allActiveGrains.length;

  // Draw main interactive canvas
  useEffect(() => {
    const canvas = displayCanvasRef.current;
    const img = loadedImageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = img.naturalWidth || img.width;
    const h = img.naturalHeight || img.height;
    canvas.width = w;
    canvas.height = h;

    // 1. Draw base image or binary mask
    if (viewMode === "mask" && maskDataUrl) {
      const maskImg = new Image();
      maskImg.onload = () => {
        ctx.fillStyle = "#090d16";
        ctx.fillRect(0, 0, w, h);
        ctx.drawImage(
          maskImg,
          crop.x * w,
          crop.y * h,
          crop.width * w,
          crop.height * h
        );
        drawOverlays(ctx, w, h);
      };
      maskImg.src = maskDataUrl;
      return;
    }

    ctx.drawImage(img, 0, 0, w, h);
    drawOverlays(ctx, w, h);

    function drawOverlays(c: CanvasRenderingContext2D, cw: number, ch: number) {
      // Darken area outside Crop ROI
      const rx = crop.x * cw;
      const ry = crop.y * ch;
      const rw = crop.width * cw;
      const rh = crop.height * ch;

      c.save();
      c.fillStyle = "rgba(2, 6, 23, 0.65)";
      // Top
      c.fillRect(0, 0, cw, ry);
      // Bottom
      c.fillRect(0, ry + rh, cw, ch - (ry + rh));
      // Left
      c.fillRect(0, ry, rx, rh);
      // Right
      c.fillRect(rx + rw, ry, cw - (rx + rw), rh);

      // ROI Border
      c.strokeStyle = activeTab === "crop" ? "#38bdf8" : "rgba(16, 185, 129, 0.7)";
      c.lineWidth = Math.max(2, Math.round(cw / 320));
      c.setLineDash(activeTab === "crop" ? [8, 6] : []);
      c.strokeRect(rx, ry, rw, rh);
      c.restore();

      if (viewMode === "original") return;

      // Draw grain markers
      const dotRadius = Math.max(4, Math.round(Math.min(cw, ch) * 0.0085));
      const fontSize = Math.max(11, Math.round(Math.min(cw, ch) * 0.017));
      c.font = `bold ${fontSize}px system-ui, sans-serif`;
      c.textAlign = "center";
      c.textBaseline = "middle";

      allActiveGrains.forEach((g, idx) => {
        const gx = g.x * cw;
        const gy = g.y * ch;
        const gr = Math.max(dotRadius * 1.4, g.radius * cw * 0.85);

        // Outer ring
        c.beginPath();
        c.arc(gx, gy, gr, 0, Math.PI * 2);
        c.strokeStyle = g.isManual
          ? "rgba(251, 191, 36, 0.9)"
          : "rgba(16, 185, 129, 0.85)";
        c.lineWidth = Math.max(1.5, cw / 450);
        c.stroke();

        if (viewMode === "numbers") {
          // Pill background for number
          c.fillStyle = "rgba(2, 6, 23, 0.78)";
          c.beginPath();
          c.arc(gx, gy, fontSize * 0.75, 0, Math.PI * 2);
          c.fill();

          c.fillStyle = g.isManual ? "#fcd34d" : "#6ee7b7";
          c.fillText(String(idx + 1), gx, gy);
        } else {
          // Center high-contrast dot
          c.beginPath();
          c.arc(gx, gy, dotRadius, 0, Math.PI * 2);
          c.fillStyle = g.isManual ? "#f59e0b" : "#10b981";
          c.fill();
          c.lineWidth = 1.5;
          c.strokeStyle = "#ffffff";
          c.stroke();
        }
      });
    }
  }, [
    imageSrc,
    allActiveGrains,
    viewMode,
    maskDataUrl,
    crop,
    activeTab,
  ]);

  // Tap-to-Edit handler on the canvas:
  // If user clicks near an existing grain marker -> remove it (-1).
  // Otherwise -> add a manual grain marker at the clicked spot (+1).
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = displayCanvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const normX = (e.clientX - rect.left) / rect.width;
    const normY = (e.clientY - rect.top) / rect.height;

    // Hit test threshold (~2.2% of image dimension)
    const hitRadius = 0.022;
    const hitRadiusSq = hitRadius * hitRadius;

    // 1. Check manual grains first
    let closestManualId: string | null = null;
    let minManualDistSq = hitRadiusSq;
    for (const mg of manualGrains) {
      const dx = mg.x - normX;
      const dy = mg.y - normY;
      const dSq = dx * dx + dy * dy;
      if (dSq < minManualDistSq) {
        minManualDistSq = dSq;
        closestManualId = mg.id;
      }
    }

    if (closestManualId) {
      setManualGrains((prev) => prev.filter((g) => g.id !== closestManualId));
      return;
    }

    // 2. Check automatic grains
    let closestAutoId: string | null = null;
    let minAutoDistSq = hitRadiusSq;
    for (const ag of visibleAutoGrains) {
      const dx = ag.x - normX;
      const dy = ag.y - normY;
      const dSq = dx * dx + dy * dy;
      if (dSq < minAutoDistSq) {
        minAutoDistSq = dSq;
        closestAutoId = ag.id;
      }
    }

    if (closestAutoId) {
      const targetId = closestAutoId;
      setRemovedAutoIds((prev) => {
        const next = new Set(prev);
        next.add(targetId);
        return next;
      });
      return;
    }

    // 3. Otherwise add a new manual grain point (+1)
    const newGrain: DetectedGrain = {
      id: `manual-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      x: normX,
      y: normY,
      radius: 0.014,
      area: 45,
      isManual: true,
    };
    setManualGrains((prev) => [...prev, newGrain]);
  };

  // File upload handler
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setSampleGroundTruth(null);
        setImageSrc(reader.result);
      }
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  // Generate sample image with chosen count
  const handleGenerateSample = (count: number) => {
    const sample = generateSampleRicePhoto(count);
    if (sample.dataUrl) {
      setSampleGroundTruth(sample.actualCount);
      setCrop({ x: 0.05, y: 0.05, width: 0.9, height: 0.9 });
      setImageSrc(sample.dataUrl);
    }
  };

  // Camera modal controls using native <dialog>
  const stopCameraStream = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    setIsCameraActive(false);
  }, []);

  const openCameraModal = async () => {
    setCameraError(null);
    const dialog = cameraDialogRef.current;
    if (dialog && !dialog.open) {
      dialog.showModal();
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      mediaStreamRef.current = stream;
      setIsCameraActive(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch (err) {
      setCameraError(
        err instanceof Error
          ? `Tidak dapat mengakses kamera: ${err.message}. Pastikan izin kamera diberikan atau gunakan tombol Unggah Foto.`
          : "Tidak dapat mengakses kamera."
      );
    }
  };

  const closeCameraModal = () => {
    stopCameraStream();
    cameraDialogRef.current?.close();
  };

  const captureFromCamera = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.95);
    setSampleGroundTruth(null);
    setImageSrc(dataUrl);
    closeCameraModal();
  };

  // Save current count to localStorage history
  const handleSaveToHistory = () => {
    const canvas = displayCanvasRef.current;
    if (!canvas) return;

    // Create small thumbnail
    const thumbCanvas = document.createElement("canvas");
    thumbCanvas.width = 160;
    thumbCanvas.height = 110;
    const tCtx = thumbCanvas.getContext("2d");
    if (tCtx) {
      tCtx.drawImage(canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
    }

    const newItem: HistoryItem = {
      id: `hist-${Date.now()}`,
      label:
        noteLabel.trim() ||
        `Sampel Beras #${history.length + 1}`,
      timestamp: new Date().toLocaleString("id-ID", {
        dateStyle: "medium",
        timeStyle: "short",
      }),
      totalCount,
      autoCount: visibleAutoGrains.length,
      manualAdded: manualGrains.length,
      manualRemoved: removedAutoIds.size,
      thumbnailUrl: thumbCanvas.toDataURL("image/jpeg", 0.7),
    };

    const updated = [newItem, ...history].slice(0, 25);
    setHistory(updated);
    setNoteLabel("");
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // Ignore quota errors
    }
  };

  const handleDeleteHistory = (id: string) => {
    const updated = history.filter((item) => item.id !== id);
    setHistory(updated);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch {
      // Ignore
    }
  };

  // Download annotated image with summary footer banner
  const handleDownloadResult = () => {
    const srcCanvas = displayCanvasRef.current;
    if (!srcCanvas) return;

    const bannerHeight = Math.max(64, Math.round(srcCanvas.height * 0.1));
    const outCanvas = document.createElement("canvas");
    outCanvas.width = srcCanvas.width;
    outCanvas.height = srcCanvas.height + bannerHeight;
    const ctx = outCanvas.getContext("2d");
    if (!ctx) return;

    // Draw image + overlays
    ctx.drawImage(srcCanvas, 0, 0);

    // Draw footer banner
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(0, srcCanvas.height, outCanvas.width, bannerHeight);

    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, srcCanvas.height);
    ctx.lineTo(outCanvas.width, srcCanvas.height);
    ctx.stroke();

    const titleFont = Math.max(16, Math.round(bannerHeight * 0.36));
    const subFont = Math.max(12, Math.round(bannerHeight * 0.22));

    ctx.fillStyle = "#10b981";
    ctx.font = `bold ${titleFont}px system-ui, sans-serif`;
    ctx.textBaseline = "middle";
    ctx.fillText(
      `Total: ${totalCount} Butir Beras`,
      24,
      srcCanvas.height + bannerHeight * 0.42
    );

    ctx.fillStyle = "#94a3b8";
    ctx.font = `${subFont}px system-ui, sans-serif`;
    const labelStr = noteLabel.trim() ? `${noteLabel.trim()} • ` : "";
    ctx.fillText(
      `${labelStr}Deteksi Otomatis: ${visibleAutoGrains.length} | Tambah Manual: +${manualGrains.length} | Dihapus: -${removedAutoIds.size} • HitungBeras CV`,
      24,
      srcCanvas.height + bannerHeight * 0.76
    );

    const link = document.createElement("a");
    link.download = `hitung-beras-${totalCount}-butir-${Date.now()}.png`;
    link.href = outCanvas.toDataURL("image/png");
    link.click();
  };

  return (
    <div className="min-h-screen bg-[#090d16] text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-slate-950/70 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/15 border border-emerald-500/40 flex items-center justify-center text-emerald-400 font-bold text-lg shadow-inner">
              🌾
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg sm:text-xl font-bold tracking-tight text-white">
                  HitungBeras
                </h1>
                <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                  Tanpa Training Model • 100% On-Device
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Deteksi & Hitung Butir Beras Otomatis (Otsu Thresholding + Distance Transform)
              </p>
            </div>
          </div>

          {/* Primary Action Buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              type="button"
              onClick={openCameraModal}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-sm transition shadow-lg shadow-emerald-950/50 cursor-pointer"
            >
              <Camera className="w-4 h-4" />
              <span>Buka Kamera</span>
            </button>

            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center gap-2 px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 font-medium text-sm transition cursor-pointer"
            >
              <Upload className="w-4 h-4 text-emerald-400" />
              <span>Unggah Foto</span>
            </button>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="hidden"
            />

            {/* Quick Sample Generator Buttons */}
            <div className="hidden md:flex items-center bg-slate-900 border border-slate-800 rounded-xl p-1 gap-1">
              <span className="text-xs text-slate-400 px-2 flex items-center gap-1">
                <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                Simulasi:
              </span>
              {[40, 65, 110].map((cnt) => (
                <button
                  key={cnt}
                  type="button"
                  onClick={() => handleGenerateSample(cnt)}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-800/90 hover:bg-emerald-500/20 hover:text-emerald-300 text-slate-300 transition cursor-pointer"
                >
                  ~{cnt} Butir
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left/Center Column: Interactive Canvas & View Controls (8 cols) */}
        <section className="lg:col-span-8 flex flex-col gap-4">
          {/* Big Counter Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-5 flex flex-wrap items-center justify-between gap-4 shadow-xl">
            <div className="flex items-center gap-4">
              <div className="px-5 py-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/30 flex flex-col items-center justify-center min-w-[120px]">
                <span className="text-xs uppercase tracking-wider font-semibold text-emerald-400">
                  Total Hitungan
                </span>
                <span className="text-4xl sm:text-5xl font-extrabold text-white tabular-nums mt-0.5">
                  {totalCount}
                </span>
                <span className="text-[11px] text-slate-400 mt-0.5">
                  butir beras
                </span>
              </div>

              <div className="space-y-1.5 text-xs sm:text-sm">
                <div className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                  <span>Deteksi Otomatis:</span>
                  <strong className="text-white tabular-nums">
                    {visibleAutoGrains.length}
                  </strong>
                  {removedAutoIds.size > 0 ? (
                    <span className="text-rose-400 text-xs">
                      (-{removedAutoIds.size} dihapus)
                    </span>
                  ) : null}
                </div>

                <div className="flex items-center gap-2 text-slate-300">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400 inline-block" />
                  <span>Koreksi Tambah Manual:</span>
                  <strong className="text-amber-300 tabular-nums">
                    +{manualGrains.length}
                  </strong>
                </div>

                <div className="flex items-center gap-3 pt-1 text-xs text-slate-400">
                  <span>Waktu proses: {processTimeMs} ms</span>
                  {sampleGroundTruth !== null ? (
                    <span className="px-2 py-0.5 rounded-md bg-slate-800 text-emerald-300 border border-slate-700">
                      Jumlah asli sampel: {sampleGroundTruth} butir
                    </span>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Export & Reset Manual Edits */}
            <div className="flex flex-wrap items-center gap-2">
              {(manualGrains.length > 0 || removedAutoIds.size > 0) && (
                <button
                  type="button"
                  onClick={() => {
                    setManualGrains([]);
                    setRemovedAutoIds(new Set());
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 transition cursor-pointer"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Reset Koreksi Manual
                </button>
              )}

              <button
                type="button"
                onClick={handleDownloadResult}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-300 border border-emerald-500/30 text-xs sm:text-sm font-medium transition cursor-pointer"
              >
                <Download className="w-4 h-4" />
                Unduh Foto Ber-label
              </button>
            </div>
          </div>

          {/* Canvas Toolbar: View Mode + Tap-to-Edit Instruction */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-3 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-xs text-slate-400 mr-1 flex items-center gap-1">
                <Eye className="w-3.5 h-3.5 text-emerald-400" />
                Tampilan:
              </span>
              {(
                [
                  { id: "dots", label: "Titik Penanda" },
                  { id: "numbers", label: "Nomor Urut" },
                  { id: "mask", label: "Masker Hitam-Putih" },
                  { id: "original", label: "Foto Asli" },
                ] as const
              ).map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => setViewMode(m.id)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-medium transition cursor-pointer ${
                    viewMode === m.id
                      ? "bg-emerald-600 text-white shadow"
                      : "bg-slate-800/80 text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1.5 text-xs text-slate-300 bg-slate-800/70 px-3 py-1.5 rounded-xl border border-slate-700/80">
              <MousePointerClick className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span>
                <strong>Tap-to-Edit:</strong> Klik area kosong untuk{" "}
                <span className="text-amber-300 font-semibold">+1</span>, klik titik untuk{" "}
                <span className="text-rose-400 font-semibold">-1</span>
              </span>
            </div>
          </div>

          {/* Interactive Canvas Container */}
          <div className="relative bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden flex items-center justify-center min-h-[340px] sm:min-h-[460px] shadow-2xl">
            <canvas
              ref={displayCanvasRef}
              onClick={handleCanvasClick}
              className="w-full h-auto max-h-[68vh] object-contain cursor-crosshair select-none"
              title="Klik pada butir beras untuk menambah/menghapus titik hitungan"
            />
          </div>

          {/* Mobile Sample Generator Bar */}
          <div className="flex md:hidden items-center justify-between bg-slate-900 border border-slate-800 rounded-xl p-2.5 gap-2">
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Coba Sampel:
            </span>
            <div className="flex gap-1.5">
              {[40, 65, 110].map((cnt) => (
                <button
                  key={cnt}
                  type="button"
                  onClick={() => handleGenerateSample(cnt)}
                  className="px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-800 text-slate-200"
                >
                  ~{cnt} Butir
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* Right Column: Parameter Sliders, Crop ROI, & History (4 cols) */}
        <aside className="lg:col-span-4 flex flex-col gap-5">
          {/* Control Panel Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-5 space-y-5">
            {/* Sub-tabs: Sensitivitas vs Crop Area */}
            <div className="grid grid-cols-2 gap-1.5 bg-slate-950 p-1 rounded-xl border border-slate-800">
              <button
                type="button"
                onClick={() => setActiveTab("sensitivity")}
                className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                  activeTab === "sensitivity"
                    ? "bg-emerald-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Sliders className="w-3.5 h-3.5" />
                Sensitivitas Deteksi
              </button>
              <button
                type="button"
                onClick={() => setActiveTab("crop")}
                className={`flex items-center justify-center gap-1.5 py-2 rounded-lg text-xs font-semibold transition cursor-pointer ${
                  activeTab === "crop"
                    ? "bg-sky-600 text-white"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Crop className="w-3.5 h-3.5" />
                Crop Area (ROI)
              </button>
            </div>

            {activeTab === "sensitivity" ? (
              <div className="space-y-4">
                {/* 1. Brightness Threshold Slider */}
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="slider-threshold" className="font-medium text-slate-200">
                      Threshold Kontras (Beras vs Alas)
                    </label>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-emerald-400 font-semibold">
                        {threshold}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          if (loadedImageRef.current) {
                            runDetection(
                              loadedImageRef.current,
                              {
                                threshold,
                                separationSensitivity,
                                minGrainSize,
                                crop,
                              },
                              true
                            );
                          }
                        }}
                        className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[11px] text-emerald-300 border border-slate-700 cursor-pointer"
                      >
                        Auto Otsu
                      </button>
                    </div>
                  </div>
                  <input
                    id="slider-threshold"
                    type="range"
                    min={30}
                    max={235}
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                  <p className="text-[11px] text-slate-400">
                    Turunkan jika ada beras agak gelap yang tidak terdeteksi; naikkan jika alas kurang gelap.
                  </p>
                </div>

                {/* 2. Touching Grain Separation Sensitivity */}
                <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="slider-separation" className="font-medium text-slate-200">
                      Pemisahan Butir Menempel
                    </label>
                    <span className="font-mono text-emerald-400 font-semibold">
                      {separationSensitivity}%
                    </span>
                  </div>
                  <input
                    id="slider-separation"
                    type="range"
                    min={5}
                    max={95}
                    value={separationSensitivity}
                    onChange={(e) => setSeparationSensitivity(Number(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                  <p className="text-[11px] text-slate-400">
                    Mengatur agresivitas algoritma <em>Distance Transform</em> saat membelah 2–4 butir beras yang saling bersentuhan.
                  </p>
                </div>

                {/* 3. Minimum Grain Area (Dust Filter) */}
                <div className="space-y-1.5 pt-2 border-t border-slate-800/80">
                  <div className="flex items-center justify-between text-xs">
                    <label htmlFor="slider-minsize" className="font-medium text-slate-200">
                      Filter Debu (Ukuran Minimum)
                    </label>
                    <span className="font-mono text-emerald-400 font-semibold">
                      {minGrainSize} px²
                    </span>
                  </div>
                  <input
                    id="slider-minsize"
                    type="range"
                    min={4}
                    max={80}
                    value={minGrainSize}
                    onChange={(e) => setMinGrainSize(Number(e.target.value))}
                    className="w-full cursor-pointer"
                  />
                  <p className="text-[11px] text-slate-400">
                    Naikkan jika bintik debu putih di atas alas ikut terhitung sebagai beras.
                  </p>
                </div>
              </div>
            ) : (
              /* Crop ROI Controls */
              <div className="space-y-4">
                <p className="text-xs text-slate-300 leading-relaxed">
                  Sesuaikan batas kotak area hitung agar pinggiran meja atau objek di luar alas gelap tidak ikut terhitung:
                </p>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div className="space-y-1">
                    <label htmlFor="crop-left" className="text-slate-300 flex justify-between">
                      <span>Batas Kiri</span>
                      <span className="font-mono text-sky-400">{Math.round(crop.x * 100)}%</span>
                    </label>
                    <input
                      id="crop-left"
                      type="range"
                      min={0}
                      max={40}
                      value={Math.round(crop.x * 100)}
                      onChange={(e) => {
                        const newX = Number(e.target.value) / 100;
                        const rightEdge = crop.x + crop.width;
                        setCrop({
                          ...crop,
                          x: newX,
                          width: Math.max(0.2, rightEdge - newX),
                        });
                      }}
                      className="w-full cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <label htmlFor="crop-right" className="text-slate-300 flex justify-between">
                      <span>Batas Kanan</span>
                      <span className="font-mono text-sky-400">
                        {Math.round((1 - (crop.x + crop.width)) * 100)}%
                      </span>
                    </label>
                    <input
                      id="crop-right"
                      type="range"
                      min={0}
                      max={40}
                      value={Math.round((1 - (crop.x + crop.width)) * 100)}
                      onChange={(e) => {
                        const rightMargin = Number(e.target.value) / 100;
                        setCrop({
                          ...crop,
                          width: Math.max(0.2, 1 - rightMargin - crop.x),
                        });
                      }}
                      className="w-full cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <label htmlFor="crop-top" className="text-slate-300 flex justify-between">
                      <span>Batas Atas</span>
                      <span className="font-mono text-sky-400">{Math.round(crop.y * 100)}%</span>
                    </label>
                    <input
                      id="crop-top"
                      type="range"
                      min={0}
                      max={40}
                      value={Math.round(crop.y * 100)}
                      onChange={(e) => {
                        const newY = Number(e.target.value) / 100;
                        const bottomEdge = crop.y + crop.height;
                        setCrop({
                          ...crop,
                          y: newY,
                          height: Math.max(0.2, bottomEdge - newY),
                        });
                      }}
                      className="w-full cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <label htmlFor="crop-bottom" className="text-slate-300 flex justify-between">
                      <span>Batas Bawah</span>
                      <span className="font-mono text-sky-400">
                        {Math.round((1 - (crop.y + crop.height)) * 100)}%
                      </span>
                    </label>
                    <input
                      id="crop-bottom"
                      type="range"
                      min={0}
                      max={40}
                      value={Math.round((1 - (crop.y + crop.height)) * 100)}
                      onChange={(e) => {
                        const bottomMargin = Number(e.target.value) / 100;
                        setCrop({
                          ...crop,
                          height: Math.max(0.2, 1 - bottomMargin - crop.y),
                        });
                      }}
                      className="w-full cursor-pointer"
                    />
                  </div>
                </div>

                <div className="flex gap-2 pt-2">
                  <button
                    type="button"
                    onClick={() => setCrop({ x: 0, y: 0, width: 1, height: 1 })}
                    className="flex-1 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 cursor-pointer"
                  >
                    Area Penuh (100%)
                  </button>
                  <button
                    type="button"
                    onClick={() => setCrop({ x: 0.1, y: 0.1, width: 0.8, height: 0.8 })}
                    className="flex-1 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-sky-300 cursor-pointer"
                  >
                    Potong Tepi 10%
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Save to Local History Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-white flex items-center gap-2">
                <History className="w-4 h-4 text-emerald-400" />
                Simpan & Riwayat Perhitungan
              </h2>
              <span className="text-xs text-slate-400">{history.length} tersimpan</span>
            </div>

            <div className="flex gap-2">
              <input
                type="text"
                value={noteLabel}
                onChange={(e) => setNoteLabel(e.target.value)}
                placeholder="Nama sampel (mis. Sampel Karung A)..."
                className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-emerald-500"
              />
              <button
                type="button"
                onClick={handleSaveToHistory}
                className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold transition cursor-pointer shrink-0"
              >
                <Save className="w-3.5 h-3.5" />
                Simpan
              </button>
            </div>

            {history.length === 0 ? (
              <p className="text-xs text-slate-500 text-center py-4">
                Belum ada riwayat perhitungan yang disimpan di perangkat ini.
              </p>
            ) : (
              <div className="space-y-2.5 max-h-64 overflow-y-auto pr-1">
                {history.map((item) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between gap-3 p-2.5 rounded-xl bg-slate-950/90 border border-slate-800/90"
                  >
                    <div className="flex items-center gap-3 min-w-0">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={item.thumbnailUrl}
                        alt={item.label}
                        className="w-12 h-9 rounded-lg object-cover border border-slate-800 shrink-0"
                      />
                      <div className="min-w-0">
                        <p className="text-xs font-semibold text-slate-200 truncate">
                          {item.label}
                        </p>
                        <p className="text-[11px] text-slate-400">{item.timestamp}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-2.5 shrink-0">
                      <div className="text-right">
                        <span className="text-sm font-bold text-emerald-400 tabular-nums">
                          {item.totalCount}
                        </span>
                        <span className="text-[10px] text-slate-400 block">butir</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteHistory(item.id)}
                        className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-slate-900 transition cursor-pointer"
                        aria-label={`Hapus ${item.label}`}
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Photography Tips Card */}
          <div className="bg-emerald-950/20 border border-emerald-500/20 rounded-2xl p-4 space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-emerald-300">
              <Info className="w-4 h-4 shrink-0" />
              Tips Foto Akurasi Tinggi (99%+)
            </div>
            <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside leading-relaxed">
              <li>
                Gunakan <strong>alas gelap polos</strong> tidak mengkilap (kertas karton hitam, kain gelap, atau buku bersampul gelap).
              </li>
              <li>
                Ratakan beras dengan jari agar <strong>tidak saling bertumpuk</strong> (bersentuhan pinggir tetap bisa dipisahkan otomatis).
              </li>
              <li>
                Ambil foto tegak lurus dari atas dengan pencahayaan merata tanpa bayangan tajam.
              </li>
            </ul>
          </div>
        </aside>
      </main>

      {/* Native Modal Dialog for Live Camera Capture per modern-web-guidance (dialog) */}
      <dialog
        ref={cameraDialogRef}
        aria-labelledby="camera-dialog-title"
        onClose={stopCameraStream}
        className="m-auto w-full max-w-xl rounded-2xl bg-slate-900 text-slate-100 border border-slate-700 p-0 shadow-2xl open:flex open:flex-col"
      >
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <h2 id="camera-dialog-title" className="text-base font-bold text-white flex items-center gap-2">
            <Camera className="w-4 h-4 text-emerald-400" />
            Ambil Foto Beras dari Kamera
          </h2>
          <button
            type="button"
            onClick={closeCameraModal}
            autoFocus
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 cursor-pointer"
            aria-label="Tutup kamera"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {cameraError ? (
            <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-500/40 text-xs text-rose-200 leading-relaxed">
              {cameraError}
            </div>
          ) : (
            <div className="relative rounded-xl overflow-hidden bg-black aspect-video flex items-center justify-center border border-slate-800">
              <video
                ref={videoRef}
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              {/* Helpful alignment frame */}
              <div className="absolute inset-6 border-2 border-dashed border-emerald-400/60 rounded-xl pointer-events-none flex items-end justify-center pb-2">
                <span className="text-[11px] bg-slate-950/80 text-emerald-300 px-2.5 py-0.5 rounded-full">
                  Posisikan alas beras di dalam kotak ini
                </span>
              </div>
            </div>
          )}

          <div className="flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={closeCameraModal}
              className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 cursor-pointer"
            >
              Batal
            </button>
            {isCameraActive && (
              <button
                type="button"
                onClick={captureFromCamera}
                className="inline-flex items-center gap-2 px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-xs font-semibold text-white shadow-lg cursor-pointer"
              >
                <CheckCircle2 className="w-4 h-4" />
                Jepret & Hitung Sekarang
              </button>
            )}
          </div>
        </div>
      </dialog>
    </div>
  );
}
