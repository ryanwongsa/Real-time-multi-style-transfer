# Real Time Multi Style Transfer in Pytorch

Implementation in Pytorch of an adaption of the paper [A LEARNED REPRESENTATION FOR ARTISTIC STYLE](https://arxiv.org/pdf/1610.07629.pdf) (Vincent Dumoulin & Jonathon Shlens & Manjunath Kudlur).

## Requirements

Conda 3 installation of python 3
```
conda create --name <env> --file requirements.txt
```

## Implementation Details

The implemtentation uses Pytorch to train a deep convolutional neural network to be able to learn multiple art styles. The code implementation might not be an exact match of the paper by Vincent Dumoulin et al. as the training details an exact loss hyperparameters were not fully described. Each feature map in the network has two weights (`alpha` and `gamma`) dedicated to each style.


## TODO

- [x] Implement the multi style transfer network
- [] Save the model to an external location
- [] remove left over training code from the notebook and put it into python files
- [] create command to train instead of using the testing notebooks.
- [] create the "transfer learning" approach described in the paper to train new styles only using the `alpha` and `gamma` weights
- [] Implement a webcam feed version
