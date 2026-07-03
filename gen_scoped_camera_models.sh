#!/bin/bash
# Generate per-drone camera-scoped Gazebo models for multi-vehicle SITL.
#
# The stock x500_depth model includes a shared OakD-Lite whose camera sensor is
# hardcoded to <topic>camera</topic>. When N vehicles spawn they all publish to
# the same flat /camera topic, so per-drone vision is impossible. This script
# emits x500_depth_<i> (+ its own OakD-Lite-<i>) variants whose RGB/depth cameras
# publish on camera_<i>/depth_camera_<i>. Spawn drone i with
# PX4_GZ_MODEL=x500_depth_<i> and bridge camera_<i> -> /px4_<i>/camera.
#
# Runs inside the PX4 container (models live in the image). Idempotent.
set -euo pipefail

N="${1:?usage: gen_scoped_camera_models.sh <num_drones>}"
M="${PX4_GZ_MODELS:-/root/PX4-Autopilot/Tools/simulation/gz/models}"

for (( i=1; i<=N; i++ )); do
  oak="$M/OakD-Lite-$i"
  x5="$M/x500_depth_$i"
  rm -rf "$oak" "$x5"

  # Per-instance camera model: rename model/link/mesh refs consistently, then
  # scope both sensor topics to this drone index.
  cp -r "$M/OakD-Lite" "$oak"
  sed -i "s/OakD-Lite/OakD-Lite-$i/g" "$oak/model.sdf" "$oak/model.config"
  sed -i "s#<topic>camera</topic>#<topic>camera_$i</topic>#" "$oak/model.sdf"
  sed -i "s#<topic>depth_camera</topic>#<topic>depth_camera_$i</topic>#" "$oak/model.sdf"

  # Per-instance airframe model that includes the scoped camera.
  cp -r "$M/x500_depth" "$x5"
  sed -i "s/name='x500_depth'/name='x500_depth_$i'/" "$x5/model.sdf"
  sed -i "s#model://OakD-Lite#model://OakD-Lite-$i#" "$x5/model.sdf"
  sed -i "s#OakD-Lite/base_link#OakD-Lite-$i/base_link#g" "$x5/model.sdf"
  sed -i "s#<name>x500_depth</name>#<name>x500_depth_$i</name>#" "$x5/model.config"

  echo "generated x500_depth_$i (camera_$i / depth_camera_$i) + OakD-Lite-$i"
done
