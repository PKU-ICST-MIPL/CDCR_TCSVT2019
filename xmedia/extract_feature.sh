echo "extracting feature..."
th extract_feature.lua \
  -data_dir data/ \
  -gpuid 1 \
  -model ./models/xmedia_20000.t7 
