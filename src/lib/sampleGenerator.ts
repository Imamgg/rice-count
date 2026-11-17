/**
 * Generates a realistic synthetic photo of rice grains spread on a dark matte background,
 * including some grains that touch at their edges, so users can test immediately.
 */
export function generateSampleRicePhoto(targetCount = 65): {
  dataUrl: string;
  actualCount: number;
} {
  const width = 960;
  const height = 640;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return { dataUrl: "", actualCount: 0 };

  // 1. Dark textured background (representing dark cardboard/cloth)
  const bgGrad = ctx.createRadialGradient(
    width * 0.5,
    height * 0.5,
    50,
    width * 0.5,
    height * 0.5,
    width * 0.7
  );
  bgGrad.addColorStop(0, "#1e293b");
  bgGrad.addColorStop(1, "#0f172a");
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, width, height);

  // Subtle surface border outside the dark sheet to demonstrate ROI Crop usefulness
  ctx.strokeStyle = "rgba(148, 163, 184, 0.18)";
  ctx.lineWidth = 4;
  ctx.strokeRect(48, 40, width - 96, height - 80);

  // Place non-overlapping (or slightly edge-touching) rice grains inside the mat
  interface PlacedGrain {
    x: number;
    y: number;
    rx: number;
    ry: number;
    angle: number;
  }

  const placed: PlacedGrain[] = [];
  const marginX = 95;
  const marginY = 80;
  let attempts = 0;

  while (placed.length < targetCount && attempts < 4000) {
    attempts++;
    const x = marginX + Math.random() * (width - marginX * 2);
    const y = marginY + Math.random() * (height - marginY * 2);
    const rx = 13 + Math.random() * 3.5; // Semi-major axis (length)
    const ry = 5.2 + Math.random() * 1.4; // Semi-minor axis (width)
    const angle = Math.random() * Math.PI;

    // Allow ~15% of grains to touch edges slightly, while preventing heavy overlap
    let overlapping = false;
    const minDist = placed.length % 7 === 0 ? 14 : 23;
    for (const other of placed) {
      const dx = x - other.x;
      const dy = y - other.y;
      if (dx * dx + dy * dy < minDist * minDist) {
        overlapping = true;
        break;
      }
    }

    if (!overlapping) {
      placed.push({ x, y, rx, ry, angle });
    }
  }

  // Draw each rice grain with realistic translucent pearl shading
  for (const g of placed) {
    ctx.save();
    ctx.translate(g.x, g.y);
    ctx.rotate(g.angle);

    // Subtle drop shadow on dark mat
    ctx.beginPath();
    ctx.ellipse(1.5, 2, g.rx, g.ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
    ctx.fill();

    // Main grain body
    ctx.beginPath();
    ctx.ellipse(0, 0, g.rx, g.ry, 0, 0, Math.PI * 2);
    const grainGrad = ctx.createRadialGradient(-g.rx * 0.25, -g.ry * 0.25, 1, 0, 0, g.rx);
    grainGrad.addColorStop(0, "#ffffff");
    grainGrad.addColorStop(0.65, "#f1f5f9");
    grainGrad.addColorStop(1, "#cbd5e1");
    ctx.fillStyle = grainGrad;
    ctx.fill();

    ctx.restore();
  }

  return {
    dataUrl: canvas.toDataURL("image/jpeg", 0.95),
    actualCount: placed.length,
  };
}
