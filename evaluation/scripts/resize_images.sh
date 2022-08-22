#!/bin/bash
for f in "$1"/*.{jpg,png}; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  convert -resize 60x60 "$f" "$1/60/${base%.*}.${base##*.}"
done
