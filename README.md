## Improving Visual Consistency for Long Video Generation (FIFO-Diffusion + VideoCrafter)

This series began with a perplexing body-flipped video ... see more in this [article](https://ycmove.github.io/2024/08/20/improving-visual-consistency.html).

<table class="center">
<thead>
    <tr>
        <th colspan="2">a person swimming in ocean, high quality, 4K resolution.</th>
    </tr>
</thead>
<tbody>
<tr>
    <td colspan="2"><img src="https://ycmove.github.io/assets/imgs/a_person_swimming_in_ocean/85-95_fifo/body_flipping.gif"/></td>
</tr>
</tbody>
</table>


## Features

- Added support for Image-to-Video (I2V) generation in FIFO-Diffusion.
- Improving Visual Consistency in the long video generation
    1. **[Seeding the initial latent frame](#seeding-the-initial-latent-frame)** as the image embedding.
    2. Use **[Weighted Q-caches](#weighted-q-caches)** in Spatio-Temporal Attention.
    3. **[Extending the Latent Uniformly](#extending-the-latent-uniformly)** before the diagonal denoising.
- More background about 3D U-net and Spatio-Temporal Attention in my [blog](https://ycmove.github.io/posts/2024-08-19-3d-u-net-in-video-diffusion-models)


### Seeding the initial latent frame
Check my [article](https://ycmove.github.io/posts/2024-08-22-trick-seeding-initial-frame) for more details.
<table class="center">
  <tr>
    <th>FIFO-Diffusion</th>
    <th>FIFO+<br>Initial Seeding</th>
    <th>FIFO+<br>Initial Seeding<br>(Autoregressive)</th>
  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_person_swimming_in_ocean/fifo_origin.gif"><img src=assets/readme/a_person_swimming_in_ocean/fifo_origin.gif ></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_person_swimming_in_ocean/fifo_origin.gif"><img src=assets/readme/a_person_swimming_in_ocean/t2v_cohe.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_person_swimming_in_ocean/TTqcache_weighted.gif"><img src=assets/readme/a_person_swimming_in_ocean/t2v_cohe_ar.gif></td>
  <tr><td style="text-align:center;" colspan="3">"a bicycle accelerating to gain speed, high quality, 4K resolution."</td>
  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_slowing_down_to_stop/fifo_origin.gif"><img src=assets/readme/a_bicycle_slowing_down_to_stop/fifo_origin.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_slowing_down_to_stop/t2v_cohe.gif"><img src=assets/readme/a_bicycle_slowing_down_to_stop/t2v_cohe.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_slowing_down_to_stop/t2v_cohe_ar.gif"><img src=assets/readme/a_bicycle_slowing_down_to_stop/t2v_cohe_ar.gif></td>
  <tr><td style="text-align:center;" colspan="3">"a bicycle slowing down to stop, high quality, 4K resolution."</td>
  </tr>
</table>


### Weighted Q-caches
Check my [article](https://ycmove.github.io/posts/2024-08-24-trick-weighted-q-caches) for more details.
<table class="center">
  <tr>
    <th>FIFO-Diffusion</th>
    <th>FIFO+Q-caches</th>

  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_person_swimming_in_ocean/fifo_origin.gif"><img src=assets/readme/a_person_swimming_in_ocean/fifo_origin.gif ></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_person_swimming_in_ocean/TTqcache_weighted.gif"><img src=assets/readme/a_person_swimming_in_ocean/TTqcache_weighted.gif ></td>
  <tr><td style="text-align:center;" colspan="2">"a person swimming in ocean, high quality, 4K resolution."</td>
  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_boat_sailing_smoothly_on_a_calm_lake/fifo_origin.gif"><img src=assets/readme/a_boat_sailing_smoothly_on_a_calm_lake/fifo_origin.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_boat_sailing_smoothly_on_a_calm_lake/TTqcache_weighted.gif"><img src=assets/readme/a_boat_sailing_smoothly_on_a_calm_lake/TTqcache_weighted.gif></td>
  <tr><td style="text-align:center;" colspan="2">"a boat sailing smoothly on a calm lake, high quality, 4K resolution."</td>
  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_leaning_against_a_tree/fifo_origin.gif"><img src=assets/readme/a_bicycle_leaning_against_a_tree/fifo_origin.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_leaning_against_a_tree/TTqcache_weighted.gif"><img src=assets/readme/a_bicycle_leaning_against_a_tree/TTqcache_weighted.gif></td>
  <tr><td style="text-align:center;" colspan="2">"a bicycle leaning against a tree, high quality, 4K resolution."</td>
  </tr>
</table>


### Extending the Latent Uniformly
Check my [article](https://ycmove.github.io/posts/2024-08-27-trick-uniform-latent) for more details.
<table class="center">
  <tr>
    <th>FIFO-Diffusion</th>
    <th>FIFO+<br>Uniform Latents</th>

  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_accelerating_to_gain_speed/fifo_origin.gif"><img src=assets/readme/a_bicycle_accelerating_to_gain_speed/fifo_origin.gif ></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_bicycle_accelerating_to_gain_speed/TTqcache_weighted.gif"><img src=assets/readme/a_bicycle_accelerating_to_gain_speed/unilatents_TTqcache_attn1_weighted90.gif ></td>
  <tr><td style="text-align:center;" colspan="2">"a bicycle accelerating to gain speed, high quality, 4K resolution."</td>
  </tr>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/unilatents_TTqcache_attn1_weighted90/fifo_origin.gif"><img src=assets/readme/a_car_stuck_in_traffic_during_rush_hour/fifo_origin.gif></td>
  <td><a href="https://github.com/YCmove/FIFO-Diffusion_public/assets/readme/a_car_stuck_in_traffic_during_rush_hour/unilatents_TTqcache_attn1_weighted90.gif"><img src=assets/readme/a_car_stuck_in_traffic_during_rush_hour/unilatents_TTqcache_attn1_weighted90.gif></td>
  <tr><td style="text-align:center;" colspan="2">"a car stuck in traffic during rush hour, high quality, 4K resolution."</td>
  </tr>
</table>

## Installation
```
conda create --name fifoplus python=3.10.14
conda activate fifoplus
pip install -r requirements.txt
```

## Downloading the Checkopoints
|Model|Resolution|Checkpoint| Config
|:----|:---------|:---------|:-------
|VideoCrafter2 (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)| [Link](https://github.com/AILab-CVC/VideoCrafter/blob/main/configs/inference_t2v_512_v2.0.yaml)
|VideoCrafter1 (Image2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt)| [Link](https://github.com/AILab-CVC/VideoCrafter/blob/main/configs/inference_i2v_512_v1.0.yaml)

Directory structure:
```
. FIFO-Diffusion_public
    ├──configs
    │     ├── inference_i2v_512_v1.0.yaml
    │     └── inference_t2v_512_v2.0.yaml
    ├──videocrafter_models
    │     ├── base_512_v2
    │     │        └── model.ckpt
    │     └── Image2Video_512
    ..             └── model.ckpt
```

## Usage

### Prompts files
For `t2v` and `t2v_seed`, the txt filw should look like
```
{prompt1}
{prompt2}
...
```
Example:
```
a person swimming in ocean, high quality, 4K resolution.
a person giving a presentation to a room full of colleagues, high quality, 4K resolution.
```

For `i2v`
```
{image_path_1};{prompt1}
{image_path_22};{prompt2}
...
```
Example:
```
/data/vbench2/a large wave crashes over a rocky cliff.jpg;a large wave crashes over a rocky cliff, high quality, 4K resolution.
/data/vbench2/A teddy bear is climbing over a wooden fence.jpg;A teddy bear is climbing over a wooden fence, high quality, 4K resolution.
```

### Argument `--mode {main_option}{sub_option}`
  - Main options:
    - `i2v`
    - `t2v`
    - `t2v_seed`: Seeding the initial latent frame
  - Sub options:
    - `TTqcache_attn1`: Enable Q-caches
    - `unilatent`: Extending the Latent Uniformly

### Argument  `--experiment {experiment}`
This will create a folder name `{experiment}` under the main directory and a `{experiment}.gif` (or mp4).
```
. FIFO-Diffusion_public
    ├──results
    ..  └── videocraft_v2_fifo
              ├── latents # this stores the clean latent from base model
              └── random_noise
                    └── {prompt}
                            └──{experiment}
```


### Inference command for main option `t2v`
```
python3 videocrafter_main.py \\
--config configs/inference_t2v_512_v2.0.yaml \\
--ckpt_path videocrafter_models/base_512_v2/model.ckpt \\
--prompt_file prompts/vbench_t2v_subject_consistency_debug.csv \\
--mode t2v_TTqcache_attn1_unilatent \\
--save_frames \\
--experiment t2v_TTqcache_attn1_unilatent
```

### Inference command for main option `i2v` and `t2v_seed`
```
python3 videocrafter_main.py \\
--config configs/inference_i2v_512_v1.0.yaml \\
--ckpt_path videocrafter_models/Image2Video_512/model.ckpt \\
--prompt_file prompts/vbench_t2v_cohe_fromi2v.csv \\
--mode t2v_seed_TTqcache \\
--save_frames \\
--experiment t2v_seed_TTqcache
```



## Acknowledgements
This repo is a fork of [FIFO-Diffusion](https://github.com/jjihwan/FIFO-Diffusion_public?tab=readme-ov-file#-citation), using [VideoCrafter](https://github.com/AILab-CVC/VideoCrafter?tab=readme-ov-file#-citation) as the base model. The ideas are also inspired by [ConsiStory: Training-Free Consistent Text-to-Image Generation](https://arxiv.org/abs/2402.03286) and [Cross-Image Attention for Zero-Shot Appearance Transfer](https://arxiv.org/abs/2311.03335). Be sure to check out and cite their original publications. And I am open to any discussions on this work!

