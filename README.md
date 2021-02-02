# Dual-Stream-PRNet-Plus  

Code for Dual-Stream PRNet++  

## build correlation_cuda  
  cd ./correlation_package  
  rm -rf *_cuda.egg-info build dist __pycache__  
  python3 setup.py install --user  

## Training  
  CUDA_VISIBLE_DEVICES=4,5 python3 launch.py --master_port 29678 train_lpba40.py --net voxnet_fusion --train_data_root ./data/LPBA40/delineation_space 
## Testing  
  python3 test_lpba.py --net voxnet_fusion --data_root ./data/LPBA40/test/ --model_file ./models/lpba40_cascade_best.pt 
