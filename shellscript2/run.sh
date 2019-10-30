for exp in 'HEPG2' 'HUVEC' 'RPE' 'U2OS'
do
  python /content/recursion-cellular-image-classification/src/train.py --cell-type $exp --gpus 0 --fp16 --with-plates
done

python /content/recursion-cellular-image-classification/src/predict.py --gpus 0 --fp16 --with-plates