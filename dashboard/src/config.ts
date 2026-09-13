// Central place for backend addressing.
// Dev (vite dev server): defaults to the Pi, override with
//   VITE_CANDLE_HOST=192.168.x.x:8000 npm run dev
// Production build: same-origin — the dashboard is served by FastAPI itself,
// so it works at whatever host:port you browsed to.
export const CANDLE_HOST: string =
  import.meta.env.VITE_CANDLE_HOST ??
  (import.meta.env.DEV ? '10.0.135.7:8000' : window.location.host);
export const API_BASE = `http://${CANDLE_HOST}`;
export const WS_STREAM_URL = `ws://${CANDLE_HOST}/ws/stream`;
