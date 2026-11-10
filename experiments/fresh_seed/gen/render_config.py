"""Track B render configuration for an existing lift_scene .blend (Blender 5.2, headless, Linux GPU).

Renders ONLY the frames the V-JEPA input pipeline uses and ONLY the pixels it uses:
  * frames 1, 1+step, ... (step 6 at 24 fps = the exact frames ffmpeg fps=4 selects: t = k/4 s);
  * render border = the centre 720x720 crop of the 1280x720 frame (prepare_vjepa_frames.sh crop),
    cropped to border, so the PNG is 720x720 and is then lanczos-scaled to 384x384 outside Blender.
Usage: blender -b scene.blend --python render_config.py -- --samples 16 --step 6 --last 427 --seed 123 --out DIR
"""
import argparse
import sys

import bpy

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument('--samples', type=int, default=16)
p.add_argument('--step', type=int, default=6)
p.add_argument('--last', type=int, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', required=True)
p.add_argument('--device', choices=['optix', 'cuda', 'cpu'], default='optix')
a, _unknown = p.parse_known_args(argv)  # profile_render.sh passes key=value extras

s = bpy.context.scene
s.render.engine = 'CYCLES'
s.cycles.samples = a.samples
s.cycles.seed = a.seed
selected = 'CPU'
if a.device != 'cpu':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for kind in ([a.device.upper(), 'CUDA'] if a.device == 'optix' else ['CUDA']):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                s.cycles.device = 'GPU'
                selected = kind + ':' + ','.join(d.name for d in gpus)
                break
        except Exception as exc:  # noqa: BLE001
            print('device %s unavailable: %s' % (kind, exc), flush=True)
# Pilot profile (ATTEMPTS.md): OIDN on CPU (4 vCPU) was the bottleneck (~6 s/frame); same OIDN denoiser on the GPU
# plus persistent data is ~0.4 s/frame. Denoiser algorithm and sample count unchanged.
if selected != 'CPU':
    s.cycles.denoising_use_gpu = True
    s.render.use_persistent_data = True
W, H, C = s.render.resolution_x, s.render.resolution_y, 720
assert (W, H) == (1280, 720), (W, H)
s.render.use_border = True
s.render.use_crop_to_border = True
s.render.border_min_x = (W - C) / 2 / W
s.render.border_max_x = (W + C) / 2 / W
s.render.border_min_y = 0.0
s.render.border_max_y = 1.0
s.frame_start = 1
s.frame_end = a.last
s.frame_step = a.step
s.render.filepath = a.out.rstrip('/') + '/frame_'
print('RENDER_DEVICE_SELECTED=%s samples=%d seed=%d step=%d last=%d denoise=%s dn_gpu=%s persist=%s denoiser=%s blender=%s' % (
    selected, a.samples, a.seed, a.step, a.last, s.cycles.use_denoising, getattr(s.cycles, 'denoising_use_gpu', None),
    s.render.use_persistent_data,
    getattr(s.cycles, 'denoiser', '?'), bpy.app.version_string), flush=True)
if a.device != 'cpu' and selected == 'CPU':
    raise RuntimeError('FATAL: GPU requested but not available')
bpy.ops.render.render(animation=True)
