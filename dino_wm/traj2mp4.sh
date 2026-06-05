#!/bin/bash

OUTPUT_DIR=${1:-/newdata2/dylantw/Legislative-Harness/dino_wm}

ffmpeg -y -framerate 12 -i "$OUTPUT_DIR/trajectory_inspect/left/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/left.mp4"

ffmpeg -y -framerate 12 -i "$OUTPUT_DIR/trajectory_inspect/right/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/right.mp4"
