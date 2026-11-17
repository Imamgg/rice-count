export interface DetectedGrain {
  id: string;
  /** Normalized X coordinate (0..1) relative to full image */
  x: number;
  /** Normalized Y coordinate (0..1) relative to full image */
  y: number;
  /** Equivalent radius in normalized width units */
  radius: number;
  /** Pixel area in processing resolution */
  area: number;
  /** Whether added manually by user via Tap-to-Edit */
  isManual?: boolean;
}

export interface CropRect {
  /** Normalized 0..1 */
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionParams {
  /** Threshold 10..245 */
  threshold: number;
  /** Separation sensitivity 1..100 (higher = splits touching grains more aggressively) */
  separationSensitivity: number;
  /** Minimum grain area in scaled pixels (filters dust/noise) */
  minGrainSize: number;
  /** Crop region (0..1) */
  crop: CropRect;
}

export interface DetectionResult {
  grains: DetectedGrain[];
  computedOtsu: number;
  maskDataUrl: string;
  processTimeMs: number;
}

/**
 * Compute Otsu's threshold on a grayscale buffer (0..255).
 */
export function computeOtsuThreshold(gray: Uint8Array): number {
  const hist = new Int32Array(256);
  const total = gray.length;
  if (total === 0) return 128;

  for (let i = 0; i < total; i++) {
    hist[gray[i]]++;
  }

  let sum = 0;
  for (let i = 0; i < 256; i++) {
    sum += i * hist[i];
  }

  let sumB = 0;
  let wB = 0;
  let wF = 0;
  let varMax = 0;
  let threshold = 128;

  for (let t = 0; t < 256; t++) {
    wB += hist[t];
    if (wB === 0) continue;
    wF = total - wB;
    if (wF === 0) break;

    sumB += t * hist[t];
    const mB = sumB / wB;
    const mF = (sum - sumB) / wF;

    const varBetween = wB * wF * (mB - mF) * (mB - mF);
    if (varBetween > varMax) {
      varMax = varBetween;
      threshold = t;
    }
  }

  return Math.max(30, Math.min(230, threshold));
}

/**
 * 3x3 Gaussian blur on grayscale image to smooth grain surface texture.
 */
function smoothGrayscale(gray: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (y === 0 || y === h - 1 || x === 0 || x === w - 1) {
        out[y * w + x] = gray[y * w + x];
        continue;
      }
      const idx = y * w + x;
      const sum =
        gray[idx - w - 1] +
        2 * gray[idx - w] +
        gray[idx - w + 1] +
        2 * gray[idx - 1] +
        4 * gray[idx] +
        2 * gray[idx + 1] +
        gray[idx + w - 1] +
        2 * gray[idx + w] +
        gray[idx + w + 1];
      out[idx] = sum >> 4;
    }
  }
  return out;
}

/**
 * Two-pass Chamfer (3,4) Distance Transform on binary mask (0 or 1).
 * Returns distance scaled by 3 (so 1 orthogonal step = 3, diagonal = 4).
 */
function chamferDistanceTransform(mask: Uint8Array, w: number, h: number): Int32Array {
  const INF = 1_000_000;
  const dist = new Int32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    dist[i] = mask[i] ? INF : 0;
  }

  // Forward pass
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const idx = row + x;
      if (dist[idx] === 0) continue;
      let d = dist[idx];
      if (x > 0) d = Math.min(d, dist[idx - 1] + 3);
      if (y > 0) {
        d = Math.min(d, dist[idx - w] + 3);
        if (x > 0) d = Math.min(d, dist[idx - w - 1] + 4);
        if (x < w - 1) d = Math.min(d, dist[idx - w + 1] + 4);
      }
      dist[idx] = d;
    }
  }

  // Backward pass
  for (let y = h - 1; y >= 0; y--) {
    const row = y * w;
    for (let x = w - 1; x >= 0; x--) {
      const idx = row + x;
      if (dist[idx] === 0) continue;
      let d = dist[idx];
      if (x < w - 1) d = Math.min(d, dist[idx + 1] + 3);
      if (y < h - 1) {
        d = Math.min(d, dist[idx + w] + 3);
        if (x < w - 1) d = Math.min(d, dist[idx + w + 1] + 4);
        if (x > 0) d = Math.min(d, dist[idx + w - 1] + 4);
      }
      dist[idx] = d;
    }
  }

  return dist;
}

/**
 * Detect and count rice grains in an HTMLImageElement or HTMLCanvasElement.
 */
export function detectRiceGrains(
  source: HTMLImageElement | HTMLCanvasElement,
  params: DetectionParams,
  autoCalculateOtsu = false
): DetectionResult {
  const t0 = performance.now();

  const srcW = source instanceof HTMLImageElement ? source.naturalWidth || source.width : source.width;
  const srcH = source instanceof HTMLImageElement ? source.naturalHeight || source.height : source.height;

  const cropX = Math.round(params.crop.x * srcW);
  const cropY = Math.round(params.crop.y * srcH);
  const cropW = Math.max(16, Math.round(params.crop.width * srcW));
  const cropH = Math.max(16, Math.round(params.crop.height * srcH));

  // Normalize processing dimension to max 640px for consistent parameters & fast response
  const MAX_DIM = 640;
  const scale = Math.min(1, MAX_DIM / Math.max(cropW, cropH));
  const w = Math.max(16, Math.round(cropW * scale));
  const h = Math.max(16, Math.round(cropH * scale));

  const workCanvas = document.createElement("canvas");
  workCanvas.width = w;
  workCanvas.height = h;
  const ctx = workCanvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    return { grains: [], computedOtsu: 128, maskDataUrl: "", processTimeMs: 0 };
  }

  ctx.drawImage(source, cropX, cropY, cropW, cropH, 0, 0, w, h);
  const imgData = ctx.getImageData(0, 0, w, h);
  const rgba = imgData.data;

  // 1. Grayscale conversion
  const rawGray = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    // Perceived luminance
    rawGray[i] = (rgba[p] * 77 + rgba[p + 1] * 150 + rgba[p + 2] * 29) >> 8;
  }

  // 2. Smooth grayscale
  const gray = smoothGrayscale(rawGray, w, h);

  // 3. Otsu threshold calculation
  const computedOtsu = computeOtsuThreshold(gray);
  const activeThreshold = autoCalculateOtsu ? computedOtsu : params.threshold;

  // 4. Binary thresholding
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    mask[i] = gray[i] >= activeThreshold ? 1 : 0;
  }

  // 5. Connected Components Labeling (8-connected)
  const visited = new Uint8Array(w * h);
  const queue = new Int32Array(w * h);

  interface Blob {
    pixels: number[];
    sumX: number;
    sumY: number;
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
  }

  const blobs: Blob[] = [];
  const minArea = Math.max(4, params.minGrainSize);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const startIdx = y * w + x;
      if (mask[startIdx] === 0 || visited[startIdx]) continue;

      let head = 0;
      let tail = 0;
      queue[tail++] = startIdx;
      visited[startIdx] = 1;

      const pixels: number[] = [];
      let sumX = 0;
      let sumY = 0;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;

      while (head < tail) {
        const curr = queue[head++];
        pixels.push(curr);
        const cy = Math.floor(curr / w);
        const cx = curr - cy * w;

        sumX += cx;
        sumY += cy;
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        // 8-neighbors
        for (let dy = -1; dy <= 1; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          const nrow = ny * w;
          for (let dx = -1; dx <= 1; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            const nIdx = nrow + nx;
            if (mask[nIdx] === 1 && !visited[nIdx]) {
              visited[nIdx] = 1;
              queue[tail++] = nIdx;
            }
          }
        }
      }

      if (pixels.length >= minArea) {
        blobs.push({ pixels, sumX, sumY, minX, maxX, minY, maxY });
      } else {
        // Erase small noise from mask so visual mask preview matches filtered grains
        for (let i = 0; i < pixels.length; i++) {
          mask[pixels[i]] = 0;
        }
      }
    }
  }

  // 6. Distance Transform for separating touching grains
  const dist = chamferDistanceTransform(mask, w, h);

  // Estimate typical single-grain area using the 40th percentile of blob areas
  // (since touching blobs are larger, the 35-45th percentile reliably represents single grains)
  const sortedAreas = blobs.map((b) => b.pixels.length).sort((a, b) => a - b);
  const refSingleArea =
    sortedAreas.length > 0
      ? sortedAreas[Math.floor(sortedAreas.length * 0.4)]
      : Math.max(25, minArea * 3);

  // Separation sensitivity (1..100): higher sensitivity = lower area ratio needed to split & smaller peak distance
  const sensFactor = (100 - params.separationSensitivity) / 100; // 0.0 (most aggressive) .. 0.99 (least aggressive)
  const splitAreaThreshold = refSingleArea * (1.25 + sensFactor * 1.15);

  const grains: DetectedGrain[] = [];
  let grainSeq = 1;

  const pushGrain = (cx: number, cy: number, area: number) => {
    // Convert workCanvas coordinates back to full image normalized 0..1 coordinates
    const fullNormX = params.crop.x + (cx / w) * params.crop.width;
    const fullNormY = params.crop.y + (cy / h) * params.crop.height;
    const eqRadiusWork = Math.sqrt(area / Math.PI);
    const fullNormRadius = (eqRadiusWork / w) * params.crop.width;

    grains.push({
      id: `auto-${grainSeq++}`,
      x: fullNormX,
      y: fullNormY,
      radius: Math.max(0.005, Math.min(0.05, fullNormRadius)),
      area,
      isManual: false,
    });
  };

  for (const blob of blobs) {
    const area = blob.pixels.length;

    // Find max distance inside this blob
    let maxDist = 0;
    for (let i = 0; i < area; i++) {
      const d = dist[blob.pixels[i]];
      if (d > maxDist) maxDist = d;
    }

    // Check if blob is likely a cluster of touching grains
    if (area >= splitAreaThreshold && maxDist >= 9) {
      // Minimum distance between two grain centers in pixels
      const avgGrainRadius = Math.max(3.5, Math.sqrt(refSingleArea / Math.PI));
      const minPeakDist = Math.max(4, avgGrainRadius * (0.85 + sensFactor * 0.9));
      const minPeakHeight = Math.max(6, maxDist * (0.38 + sensFactor * 0.25));

      // Collect candidate local maxima in distance transform within this blob
      const candidates: { idx: number; x: number; y: number; d: number }[] = [];

      for (let i = 0; i < area; i++) {
        const idx = blob.pixels[i];
        const d = dist[idx];
        if (d < minPeakHeight) continue;

        const cy = Math.floor(idx / w);
        const cx = idx - cy * w;

        let isLocalMax = true;
        for (let dy = -2; dy <= 2 && isLocalMax; dy++) {
          const ny = cy + dy;
          if (ny < 0 || ny >= h) continue;
          const nrow = ny * w;
          for (let dx = -2; dx <= 2; dx++) {
            if (dx === 0 && dy === 0) continue;
            const nx = cx + dx;
            if (nx < 0 || nx >= w) continue;
            if (dist[nrow + nx] > d) {
              isLocalMax = false;
              break;
            }
          }
        }

        if (isLocalMax) {
          candidates.push({ idx, x: cx, y: cy, d });
        }
      }

      // Sort peaks by highest distance first, then apply Non-Maximum Suppression (NMS)
      candidates.sort((a, b) => b.d - a.d);
      const acceptedPeaks: { x: number; y: number; d: number }[] = [];

      for (const cand of candidates) {
        let tooClose = false;
        for (const peak of acceptedPeaks) {
          const dx = cand.x - peak.x;
          const dy = cand.y - peak.y;
          if (dx * dx + dy * dy < minPeakDist * minPeakDist) {
            tooClose = true;
            break;
          }
        }
        if (!tooClose) {
          acceptedPeaks.push(cand);
        }
      }

      // If distance transform found multiple peaks, refine each seed's centroid via Voronoi assignment
      if (acceptedPeaks.length >= 2) {
        const seedSumX = new Float64Array(acceptedPeaks.length);
        const seedSumY = new Float64Array(acceptedPeaks.length);
        const seedCount = new Int32Array(acceptedPeaks.length);

        for (let i = 0; i < area; i++) {
          const idx = blob.pixels[i];
          const py = Math.floor(idx / w);
          const px = idx - py * w;

          let bestSeed = 0;
          let bestDistSq = Infinity;
          for (let s = 0; s < acceptedPeaks.length; s++) {
            const dx = px - acceptedPeaks[s].x;
            const dy = py - acceptedPeaks[s].y;
            const dSq = dx * dx + dy * dy;
            if (dSq < bestDistSq) {
              bestDistSq = dSq;
              bestSeed = s;
            }
          }
          seedSumX[bestSeed] += px;
          seedSumY[bestSeed] += py;
          seedCount[bestSeed]++;
        }

        for (let s = 0; s < acceptedPeaks.length; s++) {
          if (seedCount[s] >= minArea * 0.5) {
            pushGrain(
              seedSumX[s] / seedCount[s],
              seedSumY[s] / seedCount[s],
              seedCount[s]
            );
          }
        }
        continue;
      }

      // Fallback if two grains touch end-to-end forming a long smooth ridge without two distinct peaks:
      const estCount = Math.round(area / refSingleArea);
      if (estCount >= 2 && estCount <= 5) {
        const bw = blob.maxX - blob.minX + 1;
        const bh = blob.maxY - blob.minY + 1;
        for (let k = 0; k < estCount; k++) {
          const frac = (k + 0.5) / estCount;
          const gx = bw >= bh ? blob.minX + frac * bw : blob.sumX / area;
          const gy = bh > bw ? blob.minY + frac * bh : blob.sumY / area;
          pushGrain(gx, gy, area / estCount);
        }
        continue;
      }
    }

    // Single grain blob
    pushGrain(blob.sumX / area, blob.sumY / area, area);
  }

  // 7. Create visual binary mask image URL for inspection mode
  const maskImg = ctx.createImageData(w, h);
  const mData = maskImg.data;
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    const val = mask[i] ? 255 : 18;
    mData[p] = mask[i] ? 52 : 15;
    mData[p + 1] = mask[i] ? 211 : 23;
    mData[p + 2] = mask[i] ? 153 : 42;
    mData[p + 3] = 255;
    void val;
  }
  ctx.putImageData(maskImg, 0, 0);
  const maskDataUrl = workCanvas.toDataURL("image/png");

  const processTimeMs = Math.round(performance.now() - t0);

  return {
    grains,
    computedOtsu: activeThreshold,
    maskDataUrl,
    processTimeMs,
  };
}
