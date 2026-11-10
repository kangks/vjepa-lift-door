#!/bin/bash
# Render-speed profile on the exact AMI: 8 frames of anomaly_desync_tree_shadow (leaves = heaviest scene) under
# variant settings; Blender prints per-frame "Time:" lines. Usage: profile_render.sh BLENDER CODE OUTLOG
BL=$1; C=$2; OUT=$3
D=/opt/b/prof; rm -rf $D; mkdir -p $D/frames
$BL -b -t 0 --python $C/gen/lift_scene_jitter.py -- --output $D/frames --scenario anomaly_desync_tree_shadow \
  --seed 1099 --seconds 18 >/dev/null 2>&1
cat > $D/v.py <<'EOF'
import bpy, sys
a = sys.argv[sys.argv.index('--') + 1:]
s = bpy.context.scene
for kv in [x for x in a if '=' in x and not x.startswith('-')]:
    k, v = kv.split('=')
    if k == 'denoise': s.cycles.use_denoising = v == '1'
    if k == 'dgpu' and hasattr(s.cycles, 'denoising_use_gpu'): s.cycles.denoising_use_gpu = v == '1'
    if k == 'persist': s.render.use_persistent_data = v == '1'
    if k == 'adaptive': s.cycles.use_adaptive_sampling = v == '1'
    if k == 'tile': s.cycles.tile_size = int(v)
print('VARIANT', a, 'dgpu_attr', hasattr(s.cycles, 'denoising_use_gpu'),
      getattr(s.cycles, 'denoising_use_gpu', None), 'persist', s.render.use_persistent_data, flush=True)
EOF
run() {  # name samples device extra...
  name=$1; smp=$2; dev=$3; shift 3
  t0=$(date +%s)
  $BL -b $D/frames/lift_scene.blend -t 0 --python $D/v.py --python $C/gen/render_config.py -- "$@" --samples $smp \
     --step 6 --last 43 --seed 5 --out $D/out_$name --device $dev > $D/log_$name.txt 2>&1
  t1=$(date +%s)
  echo "PROFILE $name samples=$smp dev=$dev extra=[$*] wall_s=$((t1 - t0)) frames=$(ls $D/out_$name | wc -l)" | tee -a $OUT
  grep -E "VARIANT|RENDER_DEVICE|Time:" $D/log_$name.txt | head -12 >> $OUT
}
echo "== profile $(date -u +%FT%TZ)" >> $OUT
run base16 16 optix
run dgpu16 16 optix dgpu=1
run persist_dgpu16 16 optix dgpu=1 persist=1
run nodn16 16 optix denoise=0
run persist_dgpu8 8 optix dgpu=1 persist=1
run cpu16 16 cpu
