echo "extracting feature..."
th extract_feature.lua \
  -data_dir data/ \
  -save_dir ./extracted_feature_iter50000/ \
  -gpuid 3 \
  -model ./models/xmedianet_50000.t7 
  
th extract_feature.lua \
  -data_dir data/ \
  -save_dir ./extracted_feature_iter80000/ \
  -gpuid 3\
  -model ./models/xmedianet_80000.t7 
