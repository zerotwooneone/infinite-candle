//candy cane

```json
{
  "layers": [
    {
      "type": "stripes",
      "color_a": [255, 0, 0],
      "color_b": [255, 255, 255],
      "angle": 45.0,
      "width": 0.5,
      "speed": 5
    }
  ]
}
```


//candle
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


//snow
```json
{
    "layers": [
    {
        "type": "solid",
        "color": [0, 0, 1],
        "opacity": 1.0
        },
        {
        "type": "snow",
        "color": [200, 200, 255],
        "flake_count": 20,
        "gravity": 0.05,
        "wind": 0.1
    }
    ]
}
```


//lava lamp
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


//debug - vert lines
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


//fireworks
```json
{
    "layers": [
    {
        "type": "solid",
        "color": [1, 0, 1]
        },
        {
        "type": "fireworks",
        "launch_rate": 0.8,
        "burst_height": 0.7,
        "explosion_size": 0.2
    }
    ]
}
```
