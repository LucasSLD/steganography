# Installing necessary packages
```bash
>> conda env create -n env_name -f Documents/environment.yml  
```
# ImageCompressor
## Training
### Data Preparation for training of ImageCompressor model

We need to first prepare the training and validation data.
The trainging data is from flicker.com.
You can obtain the training data according to description of [CompressionData](https://github.com/liujiaheng/CompressionData)

The training details is similar to our another repo [compression](https://github.com/liujiaheng/compression)
### Running Script
```
CUDA_VISIBLE_DEVICES=0 python train.py --config examples/example/config.json -n baseline
```
## Using ImageCompressorSteganography on .pgm files
This model use the same weights obtained from the training of ImageCompressor. This model can be used to generate cover and stego images from .pgm files. You could also use .jpeg and .png but you need to write your own scripts to do so. You must convert your images to pytorch tensor and feed them to the model. The model works on pytorch tensors of shape [batch size, nb_channels, height, width] <br>See JIN_SRNet/JIN_SRNet/precover_processing.py

# SRNet for cover/stego classification
## Transfer Learning
Generate cover dataset:
```bash
python precover_processing.py -n path_to_precover_folder -o path_to_output_folder -m path_to_ImageCompressor_weights -p .0
```
Generate stego dataset (naive insertion method):
```bash
python precover_processing.py -n path_to_precover_folder -o path_to_output_folder -m path_to_ImageCompressor_weights -p 0.0004 # insertion rate of 8e-4
```
Launch the training with the following command:
```bash
python -u JIN_SRNet/LitModel/LitModel_old/train_lit_model.py --version RJCA --backbone srnet --batch-size 8 --pair-constraint 0 --lr 1e-3 --eps 1e-7 --lr-scheduler-name onecycle --optimizer-name adamax --epochs 50 --gpus 0 --weight-decay 2e-4 --decoder NR --data-path path_to_root_of_the_project/JIN_SRNet/ --seed ./JIN_SRNet/JIN_SRNet/epoch=56_val_wAUC=0.8921.pt --cover-folder-name BossBase-1.01-cover --stego-folder-name stego_001bit --payload 0_01
```
