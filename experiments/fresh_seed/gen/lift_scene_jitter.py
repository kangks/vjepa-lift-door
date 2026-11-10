"""Procedural CCTV-style lift door animation for Blender 4.x/5.x.
Run: blender -b -t 0 --python lift_scene.py -- --output /tmp/lift_frames --fps 24 --seconds 16
"""
import bpy
import math
import random
import argparse
import sys
import json
import csv
import os
from mathutils import Vector

args_raw = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='/tmp/lift_frames')
parser.add_argument('--fps', type=int, default=24)
parser.add_argument('--seconds', type=float, default=16.0)
parser.add_argument('--cycle-seconds', type=float, default=8.0, help='Door cycle length (two cycles in 16 seconds)')
parser.add_argument('--width', type=int, default=1280)
parser.add_argument('--height', type=int, default=720)
parser.add_argument('--seed', type=int, default=71)
parser.add_argument('--jitter', type=float, default=0.035, help='Maximum mechanical door jitter in metres')
parser.add_argument('--flicker', type=float, default=0.55, help='Fractional maximum light intensity dip')
parser.add_argument('--tree-shadow', type=float, default=0.85, help='Wind-blown tree shadow strength, 0 disables')
parser.add_argument('--scenario', default='healthy_clean')
parser.add_argument('--video-file', default='video.mp4')
# ---- Track B jitter extension (8 Nov 2026). All defaults 0 => identical to the original lift_scene.py.
parser.add_argument('--start-offset', type=float, default=0.0, help='door cycle start delay, s (R6: +-0.05)')
parser.add_argument('--interleaf', type=float, default=0.0, help='right-leaf timing lag, s (R6: +-0.02); healthy, not labelled')
parser.add_argument('--fault-shift', type=float, default=0.0, help='desync onset shift in nominal cycle seconds')
parser.add_argument('--flicker-shift', type=float, default=0.0, help='interior flicker burst shift, s')
parser.add_argument('--ext-shift', type=float, default=0.0, help='exterior flicker window shift, s')
parser.add_argument('--ball-shift', type=float, default=0.0, help='ball trajectory start shift, s')
parser.add_argument('--ball-lane', type=float, default=0.0, help='ball crossing lane offset, m')
parser.add_argument('--balloon-shift', type=float, default=0.0, help='balloon pass start shift, s')
parser.add_argument('--balloon-dy', type=float, default=0.0, help='balloon path vertical offset (camera space)')
parser.add_argument('--wind-phase', type=float, default=0.0, help='tree wind time offset, s')
args = parser.parse_args(args_raw)
if args.cycle_seconds <= 0:
    parser.error('--cycle-seconds must be positive')
SCENARIOS = {
 'healthy_clean': ('healthy', [], []),
 'healthy_tree_shadow': ('healthy', [], ['tree_shadow']),
 'healthy_ball_crossing': ('healthy', [], ['ball_crossing']),
 'healthy_ball_reopen': ('healthy', [], ['ball_reopen']),
 'healthy_balloon_partial': ('healthy', [], ['balloon_partial']),
 'healthy_external_flicker': ('healthy', [], ['external_flicker']),
 'anomaly_door_desync': ('anomaly', ['door_desync'], []),
 'anomaly_interior_light_flicker': ('anomaly', ['interior_light_flicker'], []),
 'anomaly_desync_tree_shadow': ('anomaly', ['door_desync'], ['tree_shadow']),
 'anomaly_desync_ball_crossing': ('anomaly', ['door_desync'], ['ball_crossing']),
 'anomaly_interior_flicker_balloon_partial': ('anomaly', ['interior_light_flicker'], ['balloon_partial']),
 'stress_balloon_full_occlusion': ('stress', [], ['balloon_full']),
}
if args.scenario not in SCENARIOS:
    parser.error('Unknown scenario: '+args.scenario)
HEALTH, FAULTS, DISTURBANCES = SCENARIOS[args.scenario]
args.jitter = 0.0  # no legacy jitter in clean scenes; only two explicit fault families
args.tree_shadow = args.tree_shadow if 'tree_shadow' in DISTURBANCES else 0.0
args.flicker = args.flicker if 'external_flicker' in DISTURBANCES else 0.0
random.seed(args.seed)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for datablocks in (bpy.data.materials, bpy.data.lights, bpy.data.cameras):
    pass
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'CPU'
scene.cycles.samples = 16
scene.render.resolution_x = args.width
scene.render.resolution_y = args.height
scene.render.resolution_percentage = 100
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = args.output.rstrip('/') + '/frame_'
scene.render.fps = args.fps
scene.frame_start = 1
scene.frame_end = round(args.fps * args.seconds)
scene.world.color = (0.025, 0.032, 0.042)
scene.render.image_settings.color_mode = 'RGB'
try:
    scene.view_settings.view_transform = 'AgX'
except Exception:
    scene.view_settings.view_transform = 'Standard'


def material(name, color, metallic=0.0, roughness=0.75, emission=0.0):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.0)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*color, 1.0)
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    if emission:
        if bsdf.inputs.get('Emission Color'):
            bsdf.inputs['Emission Color'].default_value = (*color, 1.0)
            bsdf.inputs['Emission Strength'].default_value = emission
    return m

concrete = material('dark concrete pavement', (0.19, 0.21, 0.23))
wall_mat = material('lift lobby wall', (0.35, 0.36, 0.38))
steel = material('brushed stainless door panels', (0.48, 0.54, 0.57), metallic=0.82, roughness=0.32)
trim = material('lift frame trim', (0.085, 0.095, 0.11), metallic=0.55, roughness=0.45)
inside = material('dark lift cabin interior', (0.10, 0.13, 0.16), metallic=0.18, roughness=0.65)
shadow = material('pedestrian shadow', (0.055, 0.061, 0.072), roughness=1)
leaf_mat = material('tree shadow caster leaf', (0.025, 0.045, 0.025), roughness=1)
stripes = material('street lane markings', (0.27, 0.28, 0.25))
indicator = material('lift status display', (0.05, 0.8, 0.48), emission=1.8)


def cube(name, center, scale, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new('soft edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 2
        # Weighted normals omitted for Blender 4.x compatibility on macOS.
    return obj


def ellipsoid(name, center, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=1, location=center)
    o = bpy.context.object
    o.name = name
    o.scale = radius
    o.data.materials.append(mat)
    return o

# Foreground outside lift: pavement and approximate street-light reflections.
cube('pavement', (0, -3.0, -0.10), (12, 8, 0.2), concrete)
for x in [-3.3, -0.3, 2.7]:
    cube('sidewalk line', (x, -3.4, 0.004), (0.10, 1.8, 0.009), stripes)
# Wall split around a 2.35 m wide lift opening. Opaque side walls mask pocketed doors.
# Wall front surface is y=-0.10; both moving door panels are fully BEHIND it at y=+0.05.
cube('lobby left wall', (-3.0, 0.20, 1.65), (3.66, 0.6, 3.3), wall_mat)
cube('lobby right wall', (3.0, 0.20, 1.65), (3.66, 0.6, 3.3), wall_mat)
cube('wall above door', (0, 0.20, 2.95), (2.38, 0.6, 0.7), wall_mat)
# Visible cabin interior when the doors open.
cube('cabin back wall', (0, 1.44, 1.22), (2.3, 0.12, 2.45), inside)
cube('cabin floor', (0, 0.85, -0.008), (2.3, 1.45, 0.03), inside)
cube('cabin ceiling', (0, 0.85, 2.45), (2.3, 1.45, 0.09), trim)
cube('cabin left side', (-1.15, 0.85, 1.22), (0.08, 1.4, 2.45), steel)
cube('cabin right side', (1.15, 0.85, 1.22), (0.08, 1.4, 2.45), steel)
# Door frame in front of door leaf meshes.
cube('left frame', (-1.19, -0.27, 1.26), (0.13, 0.15, 2.54), trim)
cube('right frame', (1.19, -0.27, 1.26), (0.13, 0.15, 2.54), trim)
cube('top frame', (0, -0.27, 2.52), (2.5, 0.15, 0.14), trim)
cube('threshold', (0, -0.21, 0.015), (2.4, 0.25, 0.03), steel)
left = cube('LEFT DOOR PANEL / jitter fault', (-0.575, 0.05, 1.26), (1.15, 0.065, 2.43), steel, 0.012)
right = cube('RIGHT DOOR PANEL / jitter fault', (0.575, 0.05, 1.26), (1.15, 0.065, 2.43), steel, 0.012)
# Subtle vertical panel ribs support motion visibility.
for door, sign in [(left, -1), (right, 1)]:
    rib = cube(door.name + ' edge', (sign * 0.034, 0.007, 1.26), (0.012, 0.012, 2.38), trim)
    # Parent edge strip to leaf while preserving world transform.
    rib.parent = door
    rib.matrix_parent_inverse = door.matrix_world.inverted()

cube('floor call button plate', (1.53, -0.122, 1.15), (0.19, 0.025, 0.32), trim)
cube('call button illuminated', (1.53, -0.15, 1.16), (0.058, 0.009, 0.058), indicator)

# Tree leaves (cutouts) placed between the street light and the entrance.
# They are invisible to camera rays but cast real, animated Cycles shadows
# on the steel doors and surrounding front wall. Fixed seed makes repeatable wind.
def leaf_card(name, loc, width, height, angle):
    # Plane oriented along X/Z (faces towards the door at +Y).
    verts = []
    n = 10
    for k in range(n):
        theta = 2.0 * math.pi * k / n
        variation = 0.83 + 0.18 * random.random()
        verts.append((width * 0.5 * math.cos(theta) * variation, 0.0,
                      height * 0.5 * math.sin(theta) * variation))
    mesh = bpy.data.meshes.new(name + ' mesh')
    mesh.from_pydata(verts, [], [list(range(n))])
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    ob.location = loc
    ob.rotation_euler[1] = angle
    ob.data.materials.append(leaf_mat)
    # Cycles ray-visibility property; leave shadows enabled.
    if hasattr(ob, 'visible_camera'):
        ob.visible_camera = False
    return ob

# Upper branches sit partially beyond camera view. Motion is projected onto
# the doorway by the street-side spot lamp, like leaves swaying in a breeze.
wind = bpy.data.objects.new('TREE WIND / animated foliage', None)
bpy.context.collection.objects.link(wind)
wind.location = (0.0, -1.22, 0.0)
leaf_count = 85 if args.tree_shadow > 0 else 0
for i in range(leaf_count):
    x = random.uniform(-2.55, 2.55)
    z = random.uniform(1.1, 3.7)
    leaf = leaf_card('tree leaf %03d' % i, (x, -1.22 + random.uniform(-0.12,0.12), z),
                     random.uniform(0.075, 0.25), random.uniform(0.055, 0.20),
                     random.uniform(-0.5,0.5))
    leaf.parent = wind
    leaf.matrix_parent_inverse = wind.matrix_world.inverted()

# Smooth, correlated motion plus deterministic gusts (never per-frame white noise).
for frame in range(1, scene.frame_end+1):
    t = (frame - 1) / args.fps + args.wind_phase
    gust = 0.35 * math.exp(-((t-3.0)/0.65)**2) + 0.65 * math.exp(-((t-7.2)/0.75)**2)
    wind.location.x = 0.13 * math.sin(t*1.35) + 0.075 * math.sin(t*2.6) + gust*0.18
    wind.rotation_euler[1] = 0.08*math.sin(t*1.15) + gust*0.10
    wind.rotation_euler[2] = 0.035*math.sin(t*0.8)
    wind.keyframe_insert(data_path='location', frame=frame)
    wind.keyframe_insert(data_path='rotation_euler', frame=frame)

# Lighting: stable cabin glow, flickering overhead street-side lights.
def area_light(name, xyz, power, color, size, target):
    data = bpy.data.lights.new(name, 'AREA')
    data.energy = power
    data.color = color
    data.shape = 'DISK'
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = xyz
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()
    return obj

area_light('steady lift interior', (0, 0.8, 2.3), 330, (0.74, 0.83, 1.0), 1.4, (0, 1.2, 0.6))
street_light = area_light('FLICKERING STREET LIGHT', (-1.9, -2.0, 3.1), 610, (1.0, 0.83, 0.63), 1.3, (0, -1.3, 0))
# Small-source light casts legible dappled leaf shadows on the vertical door.
tree_light = None
if args.tree_shadow > 0:
    spot = bpy.data.lights.new('TREE SHADOW STREET SUN', 'SPOT')
    spot.energy = 1350 * args.tree_shadow
    spot.color = (1.0, 0.92, 0.75)
    spot.spot_size = math.radians(85)
    spot.spot_blend = 0.35
    spot.shadow_soft_size = 0.045
    tree_light = bpy.data.objects.new('TREE SHADOW STREET SUN', spot)
    bpy.context.collection.objects.link(tree_light)
    tree_light.location = (-2.0, -4.0, 4.25)
    tree_light.rotation_euler = (Vector((0, 0.08, 1.2))-tree_light.location).to_track_quat('-Z','Y').to_euler()
street_light2 = area_light('secondary street signage', (3.0, -3.5, 2.5), 210, (0.38, 0.58, 1.0), 1.7, (0, -1.0, 1.0))

# Camera: security-camera view, facing lift with outside pavement visible.
cam_data = bpy.data.cameras.new('CCTV 01 camera')
cam = bpy.data.objects.new('CCTV 01 camera', cam_data)
bpy.context.collection.objects.link(cam)
cam.location = (2.65, -6.25, 2.9)
target = Vector((0, -0.2, 1.15))
cam.rotation_euler = (target - cam.location).to_track_quat('-Z', 'Y').to_euler()
cam_data.type = 'PERSP'
cam_data.lens = 30
scene.camera = cam




def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x*x*(3.0-2.0*x)

def lerp(a,b,u): return a+(b-a)*smoothstep(u)

def door_open_fraction(t):
    """Nominal 8-second cycle. Reopening on first cycle is a SAFE normal exception."""
    phase = t % args.cycle_seconds
    s = phase * (8.0 / args.cycle_seconds)
    if s < 0.9: return 0.0
    if s < 2.55: return smoothstep((s-.9)/1.65)
    if s < 4.1: return 1.0
    if 'ball_reopen' in DISTURBANCES and t < args.cycle_seconds:
        if s < 5.45: return 1.0 - 0.65*smoothstep((s-4.1)/1.35)
        if s < 6.15: return 0.35+0.65*smoothstep((s-5.45)/.70)
        if s < 6.7: return 1.0
        return 1.0 - smoothstep((s-6.7)/1.3)
    if s < 6.9: return 1.0-smoothstep((s-4.1)/2.8)
    return 0.0

# Physical foreground ball: visible on pavement; second trajectory crosses lift threshold.
ball = None
if 'ball_crossing' in DISTURBANCES or 'ball_reopen' in DISTURBANCES:
    ball_mat = material('bright orange rubber ball', (0.88, 0.15, 0.035), roughness=0.48)
    ball = ellipsoid('BOUNCING BALL / healthy disturbance', (-5.0,-2.25,0.25), (0.20,0.20,0.20), ball_mat)

# Camera-near balloon occluder, deliberately not an equipment fault.
balloon = None
if 'balloon_partial' in DISTURBANCES or 'balloon_full' in DISTURBANCES:
    balloon_mat = material('red balloon', (0.84, 0.035, 0.15), roughness=.28)
    balloon = ellipsoid('BALLOON / camera occlusion', (0,0,0), (0.27,0.35,0.16), balloon_mat)
    balloon.parent = cam
    balloon.location = (-2.0, 0.0, -0.7)
    if 'balloon_full' in DISTURBANCES:
        balloon.scale = (0.88, 0.80, 0.16)
    else:
        balloon.scale = (0.34, 0.42, 0.16)

# Cabin light is separate from the exterior light; only its targeted test case flickers.
cabin_light = bpy.data.objects.get('steady lift interior')
base_cabin_energy = 330.0
FS, FL = args.fault_shift, args.flicker_shift
for frame in range(1,scene.frame_end+1):
    t = (frame-1)/args.fps
    td = t - args.start_offset
    op = door_open_fraction(td)
    op_r = door_open_fraction(td - args.interleaf)
    # DESYNC: brief lag of right panel, recovery; doors otherwise remain symmetric.
    phase = (td % args.cycle_seconds) * (8.0 / args.cycle_seconds)
    ds = 0.0
    if 'door_desync' in FAULTS and 4.85+FS <= phase < 5.85+FS:
        u = (phase-4.85-FS)/1.0
        ds = 0.18 * (math.sin(math.pi*u)**2)
    left.location.x = -0.575 - 1.25*op
    right.location.x = 0.575 + 1.25*op_r + ds
    left.keyframe_insert('location', frame=frame)
    right.keyframe_insert('location', frame=frame)
    # deterministic independent interior flicker bursts (interior electrical fault)
    interior_dip=0.0
    if 'interior_light_flicker' in FAULTS and ((4.50+FL <= t < 5.70+FL) or (12.50+FL <= t < 13.70+FL)):
        interior_dip = (0.65 if (frame//3)%3==0 else (0.90 if frame%7==0 else 0.18))
    cabin_light.data.energy = base_cabin_energy*(1.0-interior_dip)
    cabin_light.data.keyframe_insert('energy',frame=frame)
    # External nuisance flicker (NOT interior fault)
    external_dip=0.0
    te = t - args.ext_shift
    if 'external_flicker' in DISTURBANCES and ((1.5<=te<2.1) or (5.8<=te<6.6) or (11.2<=te<11.8)):
        external_dip = [0.0,0.2,0.5,0.8][(frame//3)%4]
    street_light.data.energy = 610*(1-args.flicker*external_dip)
    street_light.data.keyframe_insert('energy',frame=frame)
    if ball:
        if 'ball_crossing' in DISTURBANCES:
            u=(t-2.0-args.ball_shift)/3.4
            ball.location = (-4.3+8.6*u, -2.5+args.ball_lane, 0.21+0.53*abs(math.sin(4.0*math.pi*u))) if 0<=u<=1 else (-7,-3,0.2)
        else:
            u=(t-3.7-args.ball_shift)/2.0
            ball.location = (-3.9+3.8*u, -2.8+3.1*u,0.21+0.54*abs(math.sin(3*math.pi*u))) if 0<=u<=1 else (-7,-3,.2)
        ball.keyframe_insert('location',frame=frame)
    if balloon:
        # approaches the lens halfway through the recording, including while door is moving
        u=(t-5.45-args.balloon_shift)/2.15
        balloon.location = (-2.3+3.4*u, 0.22*math.sin(8*u)+args.balloon_dy, -0.70) if 0<=u<=1 else (-4,0,-.70)
        balloon.keyframe_insert('location',frame=frame)

# Always evaluate Blender's actual keyed state for each frame.
rows=[]
for frame in range(1,scene.frame_end+1):
    scene.frame_set(frame)
    t=(frame-1)/args.fps
    td=t-args.start_offset
    op=door_open_fraction(td)
    op_r=door_open_fraction(td-args.interleaf)
    # label = explicit fault term only (healthy interleaf lag is not a fault)
    ds=right.location.x - (0.575+1.25*op_r)
    cab_dip=1.0-cabin_light.data.energy/base_cabin_energy
    ext_dip=1.0-street_light.data.energy/610
    mechanical=['door_desync'] if abs(ds)>0.005 else []
    electric=['interior_light_flicker'] if cab_dip>0.05 else []
    nuisance=[]
    bs, ls = args.ball_shift, args.balloon_shift
    if 'tree_shadow' in DISTURBANCES: nuisance.append('moving_tree_shadow')
    if 'ball_crossing' in DISTURBANCES and 2.0+bs<=t<=5.4+bs: nuisance.append('ball_crossing')
    if 'ball_reopen' in DISTURBANCES and 3.7+bs<=t<=8.0+bs: nuisance.append('ball_entered_lift_safe_reopen')
    if 'balloon_partial' in DISTURBANCES and 5.45+ls<=t<=7.6+ls: nuisance.append('balloon_partial_occlusion')
    if 'balloon_full' in DISTURBANCES and 5.45+ls<=t<=7.6+ls: nuisance.append('balloon_full_occlusion')
    if ext_dip>0.01: nuisance.append('exterior_light_flicker')
    rows.append({
       'frame':frame,'frame_index_0based':frame-1,'time_sec':round(t,6),
       'cycle_index':int(t//args.cycle_seconds),'door_openness_0_1':round(op,5),
       'left_panel_x_m':round(left.location.x,6),'right_panel_x_m':round(right.location.x,6),
       'door_sync_error_m':round(ds,6), 'interior_light_dip_fraction':round(cab_dip,5),
       'exterior_light_dip_fraction':round(ext_dip,5),
       'mechanical_fault_labels':mechanical, 'electrical_fault_labels':electric,
       'environmental_labels':nuisance, 'mechanical_anomaly':bool(mechanical),
       'anomaly':bool(mechanical or electric),
       'observable':not ('balloon_full' in DISTURBANCES and 5.65<=t<=7.2)
    })

def contiguous_events(field):
    kinds=sorted(set(v for r in rows for v in r[field]))
    events=[]
    for kind in kinds:
        start=None
        for i in range(len(rows)+1):
            active=i<len(rows) and kind in rows[i][field]
            if active and start is None: start=i
            if not active and start is not None:
                events.append({'type':kind,'start_sec':round(start/args.fps,6),
                               'end_sec_exclusive':round(i/args.fps,6),
                               'start_frame_1based':start+1,'end_frame_1based_inclusive':i})
                start=None
    return events

from pathlib import Path
out=Path(args.output).resolve().parent
out.mkdir(parents=True,exist_ok=True)
metadata={
 'schema_version':'2.0', 'generator':'procedural Blender 4.x/5.x',
 'video_file':args.video_file, 'scenario':args.scenario,
 'video_health': HEALTH, 'fault_families':FAULTS,'disturbance_types':DISTURBANCES,
 'health_definition':'Only mechanical door desync and cabin light flicker count as equipment anomalies; obstructions cause safe reopen',
 'stress_definition':'full balloon occlusion has unknown equipment observability; do not score as confident healthy detection',
 'fps':args.fps,'frame_count':scene.frame_end,'duration_sec':scene.frame_end/args.fps,
 'resolution':[args.width,args.height],'seed':args.seed,
 'trackb_jitter':{k:getattr(args,k) for k in ('cycle_seconds','start_offset','interleaf','fault_shift','flicker_shift','ext_shift','ball_shift','ball_lane','balloon_shift','balloon_dy','wind_phase')},
 'time_convention':'time_sec=(1-based-frame-1)/fps; event ends exclusive',
 'event_intervals':contiguous_events('mechanical_fault_labels')+contiguous_events('electrical_fault_labels')+contiguous_events('environmental_labels'),
 'frames':rows
}
(out/'ground_truth.json').write_text(json.dumps(metadata,indent=2))
with (out/'ground_truth_frames.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
    for r in rows:
        item=r.copy()
        for field in ('mechanical_fault_labels','electrical_fault_labels','environmental_labels'):
            item[field]=';'.join(item[field])
        writer.writerow(item)
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.output)/'lift_scene.blend'))
print('SCENARIO=%s HEALTH=%s frames=%d' % (args.scenario, HEALTH,scene.frame_end))
