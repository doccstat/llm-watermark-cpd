#!/bin/bash

for method in gumbel transform; do
  mkdir -p seedbs-not-textgen/opt-watermark200-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark100-nowatermark100-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark80-nowatermark60-watermark60-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.pfacebook.opt-1.3b$method
done

for method in gumbel transform; do
  mkdir -p seedbs-not-textgen/gpt-watermark200-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark100-nowatermark100-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark80-nowatermark60-watermark60-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.popenai-community.gpt2$method
done

chmod +x ./detect.sh
parallel -j 50 --progress ./detect.sh {1} {2} {3} ::: gumbel transform ::: $(seq 1 100) ::: watermark200 watermark100-nowatermark100 watermark80-nowatermark60-watermark60 watermark40-nowatermark50-watermark60-nowatermark20-watermark30
