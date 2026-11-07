export interface BaseLayer {
  type: string;
  opacity?: number;
  h_min?: number;
  h_max?: number;
  id: string; // for UI tracking
}

export interface SolidLayer extends BaseLayer {
  type: 'solid';
  color: [number, number, number];
}

export interface ChaseLayer extends BaseLayer {
  type: 'chase';
  color: [number, number, number];
  speed: number;
  tail_length: number;
}

export interface StripesLayer extends BaseLayer {
  type: 'stripes';
  color_a: [number, number, number];
  color_b: [number, number, number];
  angle: number;
  width: number;
  speed: number;
}

export interface SnowLayer extends BaseLayer {
  type: 'snow';
  color: [number, number, number];
  flake_count: number;
  gravity: number;
}

export interface FireLayer extends BaseLayer {
  type: 'fire';
  cooling: number;
  sparking: number;
  color_start: [number, number, number];
  color_end: [number, number, number];
}

export interface FireworkLayer extends BaseLayer {
  type: 'fireworks';
  launch_rate: number;
  burst_height: number;
  explosion_size: number;
  max_rockets: number;
  max_sparks: number;
  spark_density: number;
  rocket_speed: number;
  rocket_wiggle: number;
  rocket_gravity: number;
  spark_gravity: number;
  spark_drag: number;
  trail_decay: number;
  brightness: number;
}

export interface GameOfLifeLayer extends BaseLayer {
  type: 'gol';
  color: [number, number, number];
  bg_color: [number, number, number];
  speed: number;
  transparent: boolean;
}

export interface LavaLampLayer extends BaseLayer {
  type: 'lava';
  color: [number, number, number];
  bg_color: [number, number, number];
  blob_count: number;
  speed: number;
}

export interface BreathingLayer extends BaseLayer {
  type: 'breathing';
  color_a: [number, number, number];
  color_b: [number, number, number];
  speed: number;
}

export interface ClipLayer extends BaseLayer {
  type: 'clip';
  filename: string;
  transparent: boolean;
}

export interface AlienLayer extends BaseLayer {
  type: 'alien';
  ship_color_1: [number, number, number];
  ship_color_2: [number, number, number];
  beam_color: [number, number, number];
  speed: number;
  transparent: boolean;
}

export interface ChristmasTreeLayer extends BaseLayer {
  type: 'christmas_tree';
  rotate_speed: number;
  brightness: number;
  thickness: number;
  ornament_count: number;
  ornament_size: number;
  tree_color: [number, number, number];
  star_color: [number, number, number];
  ornament_palette: [number, number, number][];
  star_height: number;
}

export interface PluginParamSpec {
  name: string;
  label: string;
  kind: 'float' | 'int' | 'color' | 'bool' | 'choice';
  default: any;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options?: string[] | null;
  live: boolean;
}

export interface PluginDefinition {
  id: string;
  name: string;
  description?: string;
  params: PluginParamSpec[];
  defaults: Record<string, any>;
}

export interface DynamicPluginLayer {
  id: string; // for UI tracking
  type: string;
  [key: string]: any;
}

export type LayerConfig =
  | SolidLayer
  | ChaseLayer
  | StripesLayer
  | SnowLayer
  | FireLayer
  | FireworkLayer
  | GameOfLifeLayer
  | LavaLampLayer
  | BreathingLayer
  | ClipLayer
  | AlienLayer
  | ChristmasTreeLayer
  | DynamicPluginLayer;

export interface SceneRequest {
  layers: LayerConfig[];
}
