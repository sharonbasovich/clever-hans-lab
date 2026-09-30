import { IMG_PIXELS, IMG_SIZE, type World } from '../world.ts';

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | ((e: Event) => void)> = {},
  ...children: (Node | string)[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (typeof v === 'function') node.addEventListener(k, v as EventListener);
    else if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) node.append(c);
  return node;
}

export function clear(node: HTMLElement): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function pct(x: number, digits = 1): string {
  return (x * 100).toFixed(digits) + '%';
}

export function announce(msg: string): void {
  const live = document.getElementById('aria-live');
  if (live) live.textContent = msg;
}

// Paint one world image into a canvas, scaled up with crisp pixels.
export function drawImage(canvas: HTMLCanvasElement, world: World, index: number, scale = 4): void {
  const tmp = document.createElement('canvas');
  tmp.width = IMG_SIZE;
  tmp.height = IMG_SIZE;
  const ctx = tmp.getContext('2d')!;
  const data = ctx.createImageData(IMG_SIZE, IMG_SIZE);
  const off = index * IMG_PIXELS * 3;
  for (let p = 0; p < IMG_PIXELS; p++) {
    data.data[p * 4 + 0] = Math.round(world.images[off + p * 3] * 255);
    data.data[p * 4 + 1] = Math.round(world.images[off + p * 3 + 1] * 255);
    data.data[p * 4 + 2] = Math.round(world.images[off + p * 3 + 2] * 255);
    data.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  canvas.width = IMG_SIZE * scale;
  canvas.height = IMG_SIZE * scale;
  const c2 = canvas.getContext('2d')!;
  c2.imageSmoothingEnabled = false;
  c2.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

// Draw one image with the background greyed out ('noBg') or the shape erased
// into the mean background colour ('noFg') — the counterfactual ablation view.
export function drawAblation(
  canvas: HTMLCanvasElement,
  world: World,
  index: number,
  mode: 'noBg' | 'noFg',
  scale = 8,
): void {
  const off = index * IMG_PIXELS * 3;
  const mOff = index * IMG_PIXELS;
  const img = new Float32Array(world.images.subarray(off, off + IMG_PIXELS * 3));
  if (mode === 'noFg') {
    let r = 0, g = 0, b = 0, c = 0;
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (world.fgMasks[mOff + p] === 0) {
        r += img[p * 3];
        g += img[p * 3 + 1];
        b += img[p * 3 + 2];
        c++;
      }
    }
    if (c) {
      r /= c;
      g /= c;
      b /= c;
    }
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (world.fgMasks[mOff + p] === 1) {
        img[p * 3] = r;
        img[p * 3 + 1] = g;
        img[p * 3 + 2] = b;
      }
    }
  } else {
    for (let p = 0; p < IMG_PIXELS; p++) {
      if (world.fgMasks[mOff + p] === 0) {
        img[p * 3] = 0.5;
        img[p * 3 + 1] = 0.5;
        img[p * 3 + 2] = 0.5;
      }
    }
  }
  const tmp = document.createElement('canvas');
  tmp.width = IMG_SIZE;
  tmp.height = IMG_SIZE;
  const ctx = tmp.getContext('2d')!;
  const data = ctx.createImageData(IMG_SIZE, IMG_SIZE);
  for (let p = 0; p < IMG_PIXELS; p++) {
    data.data[p * 4] = Math.round(img[p * 3] * 255);
    data.data[p * 4 + 1] = Math.round(img[p * 3 + 1] * 255);
    data.data[p * 4 + 2] = Math.round(img[p * 3 + 2] * 255);
    data.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
  canvas.width = IMG_SIZE * scale;
  canvas.height = IMG_SIZE * scale;
  const c2 = canvas.getContext('2d')!;
  c2.imageSmoothingEnabled = false;
  c2.drawImage(tmp, 0, 0, canvas.width, canvas.height);
}

export function describeImage(world: World, index: number): string {
  const shape = world.labels[index] === 0 ? 'circle' : 'triangle';
  const cue =
    world.cue[index] === -1
      ? 'a grey background'
      : world.cue[index] === 0
        ? world.cueType === 'dot'
          ? 'a yellow corner dot'
          : 'a red background'
        : world.cueType === 'dot'
          ? 'a magenta corner dot'
          : 'a blue background';
  return `Training image ${index + 1}: a white ${shape} on ${cue}`;
}

// Draw image + occlusion heat overlay + foreground outline.
export function drawHeatmap(
  canvas: HTMLCanvasElement,
  world: World,
  index: number,
  heat: Float32Array,
  scale = 8,
): void {
  drawImage(canvas, world, index, scale);
  const ctx = canvas.getContext('2d')!;
  let max = 0;
  for (let i = 0; i < IMG_PIXELS; i++) max = Math.max(max, heat[i]);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const h = max > 0 ? heat[y * IMG_SIZE + x] / max : 0;
      const fg = world.fgMasks[index * IMG_PIXELS + y * IMG_SIZE + x] === 1;
      for (let sy = 0; sy < scale; sy++) {
        for (let sx = 0; sx < scale; sx++) {
          const px = ((y * scale + sy) * canvas.width + (x * scale + sx)) * 4;
          // Heat: orange glow proportional to normalized heat.
          img.data[px] = Math.min(255, img.data[px] + Math.round(180 * h));
          img.data[px + 1] = Math.round(img.data[px + 1] * (1 - 0.4 * h));
          img.data[px + 2] = Math.round(img.data[px + 2] * (1 - 0.5 * h));
          if (fg) {
            img.data[px] = Math.round(img.data[px] * 0.9);
            img.data[px + 1] = Math.min(255, img.data[px + 1] + 40);
          }
        }
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  // Foreground outline: mark fg pixels bordering a bg pixel.
  ctx.strokeStyle = '#d4ff4f';
  ctx.lineWidth = Math.max(1, scale / 4);
  ctx.beginPath();
  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const p = y * IMG_SIZE + x;
      if (world.fgMasks[index * IMG_PIXELS + p] !== 1) continue;
      const edge =
        (x > 0 && world.fgMasks[index * IMG_PIXELS + p - 1] === 0) ||
        (x < IMG_SIZE - 1 && world.fgMasks[index * IMG_PIXELS + p + 1] === 0) ||
        (y > 0 && world.fgMasks[index * IMG_PIXELS + p - IMG_SIZE] === 0) ||
        (y < IMG_SIZE - 1 && world.fgMasks[index * IMG_PIXELS + p + IMG_SIZE] === 0);
      if (edge) ctx.rect(x * scale, y * scale, scale, scale);
    }
  }
  ctx.stroke();
}

// Simple line chart of training loss (and accuracy on a second axis).
export function drawLossChart(
  canvas: HTMLCanvasElement,
  losses: number[],
  accs: number[],
  reducedMotion: boolean,
): void {
  const w = (canvas.width = canvas.clientWidth || 520);
  const h = (canvas.height = 200);
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, w, h);
  const pad = 34;
  const maxL = Math.max(0.7, ...losses);
  const n = Math.max(2, losses.length);
  const xOf = (i: number) => pad + (i / (n - 1)) * (w - pad * 2);
  const yL = (v: number) => h - pad - (v / maxL) * (h - pad * 2);
  const yA = (v: number) => h - pad - v * (h - pad * 2);

  ctx.strokeStyle = 'rgba(148,163,184,0.35)';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const y = pad + (g / 4) * (h - pad * 2);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  ctx.fillStyle = '#94a3b8';
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText('loss', pad - 6, pad - 8);
  ctx.fillText('acc', w - pad + 4, pad - 8);
  ctx.fillText('epoch →', w - pad - 44, h - 8);
  for (let i = 0; i < losses.length; i++) ctx.fillText(String(i + 1), xOf(i) - 3, h - pad + 14);

  const draw = (vals: number[], yOf: (v: number) => number, color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = xOf(i);
      const y = yOf(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    if (!reducedMotion) {
      ctx.fillStyle = color;
      vals.forEach((v, i) => {
        ctx.beginPath();
        ctx.arc(xOf(i), yOf(v), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  };
  draw(losses, yL, '#f87171');
  draw(accs, yA, '#4ade80');
}
