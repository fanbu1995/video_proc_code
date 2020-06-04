#!/bin/bash
export PATH="$PATH:/srv/apps/ffmpeg"
for f in *.LRV; do ffmpeg -i $f -vn -ac 1 -ar 16000 -acodec pcm_s16le -map 0:a:0 "${f%.*}.wav"; done
mv *.wav ~/mount/filetransfer/audio/
