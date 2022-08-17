#!/bin/sh

arg=$1;
counter = 900

resize() {
  for x in $1/*; do
    if [-d "$x" ]; then
      resize $x;
    else
      counter = counter + 1
      outname=$(printf "%05d.jpg" $counter)
      convert -crop 1360X800+276+320 "$1" "$outname"
}
