echo 'TRAINING START...........'
sh train.sh
echo 'EXTRACTION START...........'
sh extract_feature.sh
echo 'EVALUATION START...........'
matlab -nodesktop -nosplash -r "run calMAP/evalMAP;quit;"