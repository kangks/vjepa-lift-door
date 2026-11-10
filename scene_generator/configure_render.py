"""Configure Cycles for an existing .blend file, with optional Apple Metal acceleration."""
import argparse
import bpy
import sys

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
p = argparse.ArgumentParser()
p.add_argument('--device', choices=['auto', 'metal', 'cpu'], default='auto')
p.add_argument('--samples', type=int, default=16)
p.add_argument('--render', action='store_true')
args = p.parse_args(argv)
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = args.samples
scene.cycles.device = 'CPU'
selected = 'CPU'

if args.device != 'cpu' and sys.platform == 'darwin':
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        devices = list(prefs.devices)
        metals = [d for d in devices if d.type == 'METAL']
        if metals:
            for device in devices:
                device.use = (device.type == 'METAL')
            scene.cycles.device = 'GPU'
            selected = 'METAL'
        else:
            print('No Metal Cycles device found; using CPU.')
    except Exception as exc:
        print('Metal configuration unavailable (%s); using CPU.' % exc)
elif args.device == 'metal':
    print('Metal requested on a non-macOS host; using CPU.')

# Save the render configuration to the existing scene. The Metal device preference
# can be session-specific, so the caller also uses this Python file on render.
print('RENDER_DEVICE_SELECTED=' + selected, flush=True)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
if args.render:
    bpy.ops.render.render(animation=True)
