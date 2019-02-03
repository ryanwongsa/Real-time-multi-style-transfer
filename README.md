# Real Time Multi Style Transfer in Pytorch

Implementation in Pytorch of an adaption of the paper [A LEARNED REPRESENTATION FOR ARTISTIC STYLE](https://arxiv.org/pdf/1610.07629.pdf) (Vincent Dumoulin & Jonathon Shlens & Manjunath Kudlur).

## Requirements

Conda 3 installation of python 3
```
conda create --name <env> --file requirements.txt
```

## Implementation Details

The implemtentation uses Pytorch to train a deep convolutional neural network to be able to learn multiple art styles. The code implementation might not be an exact match of the paper by Vincent Dumoulin et al. as the training details an exact loss hyperparameters were not fully described. Each feature map in the network has two weights (`alpha` and `gamma`) dedicated to each style.

## Usage

Download the training dataset, using the coco dataset but can use any image dataset:
```
sh dataset/download_coco.sh
```

Training from scratch:
```
python trainstylenet.py --dataset-images {dataset_images} --styles-dir {styles_dir} --num-workers {num_workers} --model-dir {model_dir} --eval-image-dir {eval_image_dir} 
```

Inference
```
python inferstylenet.py --input-image ../bird.jpg --model-save-dir style16/pastichemodel-FINAL.pth --style-choice 4 --style-choice-2 5 --style-factor 0.5
```


## TODO

- [x] Implement the multi style transfer network
- [ ] Save the model to an external location
- [x] remove left over training code from the notebook and put it into python files
- [x] create command to train instead of using the testing notebooks.
- [x] create the "transfer learning" approach described in the paper to train new styles only using the `alpha` and `gamma` weights
- [x] Apply fine-tune approach
- [x] Implement a webcam feed version
- [x] refactor training code into class
- [x] Fix bug with resblock
- [x] Document code
- [ ] Fix windows only dataloader problem