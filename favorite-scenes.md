## alien
```json
{
  "layers": [
    {
      "type": "solid",
      "color": [0, 0, 1]
    },
    {
      "type": "alien",
      "speed": 1.0,
      "transparent": true,
      "ship_color_1": [50, 150, 255],
      "ship_color_2": [0, 255, 50],
      "beam_color": [100, 255, 255]
    }
  ]
}
```

## fireworks 1
```json
{
    "layers": [
      {
        "type": "fireworks",
        "opacity": 1.0,
        "h_min": 0.0,
        "h_max": 1.0,

        "launch_rate": 1.2,
        "burst_height": 0.78,
        "explosion_size": 0.02,

        "max_rockets": 2,
        "max_sparks": 400,
        "spark_density": 0.8,

        "rocket_speed": 0.99,
        "rocket_wiggle": 0.02,
        "rocket_gravity": -0.35,

        "spark_gravity": -0.85,
        "spark_drag": 0.50,

        "trail_decay": 2.8,
        "brightness": 1.0
      }
    ]
  }
```

## candy cane
```json
{
  "layers": [
    {
      "type": "stripes",
      "color_a": [55, 0, 0],
      "color_b": [0, 55, 0],
      "angle": 45.0,
      "width": 0.5,
      "speed": 1
    },
    {
      "type": "clip",
      "filename": "snow_heavy.npy",
      "transparent": true
    }
  ]
}
```


## candle
```json
{
    "layers": [
    {
        "type": "solid",
        "color": [50, 45, 40],
        "h_min": 0.0,
        "h_max": 0.55
        },
        {
        "type": "fire",
        "h_min": 0.55,
        "h_max": 1.0,
        "color_start": [255, 180, 40],
        "color_end": [255, 0, 0],
        "cooling": 3.2,
        "sparking": 0.5
    }
    ]
}
```


## snow
```json
{
  "layers": [
    {
      "type": "solid",
      "color": [1, 0, 2]
    },
    {
      "type": "clip",
      "filename": "snow_heavy.npy",
      "transparent": true
    }
  ]
}
```


## lava lamp
```json
{
    "layers": [
    {
        "type": "lava",
        "color": [255, 0, 0],      
        "bg_color": [0, 200, 0],  
        "blob_count": 6,
        "speed": 0.8,
        "opacity": 1.0
    }
    ]
}
```


## debug - vert lines
```json
{
    "layers": [
    {
        "type": "stripes",
        "color_a": [255, 0, 0],
        "color_b": [0, 0, 0],
        "angle": 90.0,
        "width": 0.1,
        "speed": 0.0
    }
    ]
}
```


## fireworks
```json
{
  "layers": [
    {
      "type": "solid",
      "color": [1, 0, 2]
    },
    {
      "type": "clip",
      "filename": "fireworks_show.npy",
      "transparent": true
    }
  ]
}
```

## Geometric Wireframes
```json
{
  "layers": [
    {
      "type": "solid",
      "color": [10, 0, 20]
    },
    {
      "type": "clip",
      "filename": "wireframes_v1.npy",
      "transparent": true
    }
  ]
}
```

## Plasma
```json
{
  "layers": [
    {
      "type": "solid",
      "color": [0, 0, 30]
    },
    {
      "type": "clip",
      "filename": "plasma_rainbow.npy",
      "transparent": true,
      "opacity": 1.0
    }
  ]
}
```

## chrome
```json
{
  "layers": [
    {
      "type": "clip",
      "filename": "chrome_spin.npy"
    }
  ]
}
```

## rubik's cube
```json
{
  "layers": [
    
    {
      "type": "clip",
      "filename": "rubiks_solve_loop.npy",
      "transparent": true
    }
  ]
}
```


