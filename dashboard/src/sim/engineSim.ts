import { LayerConfig } from '../types/schemas';

/**
 * Lightweight procedural simulator for dev mode.
 * Produces the same (600*3) RGB byte frames the WebSocket delivers,
 * so the digital twin consumes it through the identical code path.
 *
 * This is an approximation — it mirrors each effect's visual intent,
 * not its exact math.
 */

const LED_COUNT = 600;

// Deterministic pseudo-noise (cheap, no deps)
function noise1(x: number): number {
  const s = Math.sin(x * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function blendInto(
  out: Uint8Array,
  i: number,
  r: number, g: number, b: number,
  opacity: number,
) {
  const o = i * 3;
  out[o]     = out[o]     * (1 - opacity) + r * opacity;
  out[o + 1] = out[o + 1] * (1 - opacity) + g * opacity;
  out[o + 2] = out[o + 2] * (1 - opacity) + b * opacity;
}

export function simulateFrame(
  layers: LayerConfig[],
  t: number,
  coordsX: number[],
  coordsY: number[],
  out: Uint8Array,
): void {
  out.fill(0);

  for (const layer of layers) {
    const op = layer.opacity ?? 1.0;
    const hMin = layer.h_min ?? 0.0;
    const hMax = layer.h_max ?? 1.0;
    const l = layer as any;

    for (let i = 0; i < LED_COUNT; i++) {
      const h = coordsY[i];
      if (h < hMin || h > hMax) continue;
      const a = coordsX[i];

      switch (layer.type) {
        case 'solid': {
          const [r, g, b] = l.color ?? [0, 0, 0];
          blendInto(out, i, r, g, b, op);
          break;
        }

        case 'breathing': {
          const k = 0.5 + 0.5 * Math.sin(t * (l.speed ?? 0.5) * Math.PI * 2);
          const [ar, ag, ab] = l.color_a ?? [0, 0, 0];
          const [br, bg, bb] = l.color_b ?? [255, 255, 255];
          blendInto(out, i, lerp(ar, br, k), lerp(ag, bg, k), lerp(ab, bb, k), op);
          break;
        }

        case 'stripes': {
          const angle = ((l.angle ?? 0) * Math.PI) / 180;
          const pos = a * Math.cos(angle) + h * Math.sin(angle) * 2.28;
          const phase = (pos + t * (l.speed ?? 1) * 0.1) % 1;
          const inStripe = phase < (l.width ?? 0.2);
          const c = inStripe ? (l.color_a ?? [255, 255, 255]) : (l.color_b ?? [0, 0, 0]);
          blendInto(out, i, c[0], c[1], c[2], op);
          break;
        }

        case 'fire': {
          // Flicker: per-LED noise + height gradient between color_start/end
          const n = noise1(i * 0.37 + Math.floor(t * 20) * 0.61);
          const heat = Math.max(0, (1 - h) * (0.6 + 0.4 * n) - (l.cooling ?? 0.15) * 0.3);
          const [sr, sg, sb] = l.color_start ?? [255, 0, 0];
          const [er, eg, eb] = l.color_end ?? [255, 255, 0];
          blendInto(out, i, lerp(sr, er, heat), lerp(sg, eg, heat), lerp(sb, eb, heat), op);
          break;
        }

        case 'snow': {
          // Deterministic falling flakes
          const count = l.flake_count ?? 40;
          const speed = l.gravity ?? 0.05;
          let lit = false;
          for (let f = 0; f < count; f++) {
            const fa = noise1(f * 7.13);
            const fy = 1.0 - ((noise1(f * 3.71) + t * speed * (0.5 + noise1(f))) % 1.0);
            if (Math.abs(fy - h) < 0.01 && Math.abs(fa - a) < 0.02) { lit = true; break; }
          }
          if (lit) {
            const [r, g, b] = l.color ?? [200, 200, 255];
            blendInto(out, i, r, g, b, op);
          }
          break;
        }

        case 'lava': {
          const blobs = l.blob_count ?? 6;
          let intensity = 0;
          for (let bl = 0; bl < blobs; bl++) {
            const ba = noise1(bl * 5.3);
            const bh = 0.5 + 0.45 * Math.sin(t * (l.speed ?? 0.5) + bl * 2.1);
            const d = Math.hypot((a - ba) * 1.0, (h - bh) * 2.28);
            intensity = Math.max(intensity, Math.max(0, 1 - d * 8));
          }
          const [r, g, b] = l.color ?? [255, 0, 0];
          const [br, bg, bb] = l.bg_color ?? [20, 0, 0];
          blendInto(out, i, lerp(br, r, intensity), lerp(bg, g, intensity), lerp(bb, b, intensity), op);
          break;
        }

        case 'chase': {
          const tail = l.tail_length ?? 30;
          const head = Math.floor(t * (l.speed ?? 20)) % LED_COUNT;
          const d = (head - i + LED_COUNT) % LED_COUNT;
          if (d < tail) {
            const fade = 1 - d / tail;
            const [r, g, b] = l.color ?? [255, 255, 255];
            blendInto(out, i, r * fade, g * fade, b * fade, op);
          }
          break;
        }

        case 'gol': {
          // Blinking pseudo-cellular pattern
          const cell = Math.floor(a * 20) + Math.floor(h * 60) * 20;
          const gen = Math.floor(t * (l.speed ?? 10));
          const alive = noise1(cell * 3.7 + gen * 13.1) > 0.6;
          const c = alive ? (l.color ?? [0, 255, 0]) : (l.bg_color ?? [0, 0, 0]);
          if (alive || !l.transparent) blendInto(out, i, c[0], c[1], c[2], op);
          break;
        }

        case 'fireworks': {
          // Occasional radial bursts
          const cycle = Math.floor(t * (l.launch_rate ?? 0.5));
          const phase = (t * (l.launch_rate ?? 0.5)) % 1;
          const ba = noise1(cycle * 9.1);
          const bh = (l.burst_height ?? 0.8);
          const radius = phase * (l.explosion_size ?? 0.15) * 4;
          const d = Math.hypot((a - ba), (h - bh) * 2.28);
          const ring = Math.abs(d - radius) < 0.02 && phase > 0.3;
          if (ring) {
            const fade = 1 - phase;
            blendInto(out, i, 255 * fade, 200 * fade, 100 * fade, op * (l.brightness ?? 1));
          }
          break;
        }

        case 'clip':
        case 'alien':
        case 'christmas_tree':
        default: {
          // Can't simulate baked/complex content — dim ambient pulse as a placeholder
          const pulse = 8 + 6 * Math.sin(t * 2 + h * 4);
          blendInto(out, i, pulse, pulse, pulse * 1.5, op * 0.5);
          break;
        }
      }
    }
  }
}
