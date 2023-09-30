# Image Recognition with PyTorch

## Project Description
This PoC aims to develop a machine learning model for image recognition. We will be using PyTorch for model development and training.

## Requirements
- Python 3.x
- Pip
- Virtualenv

## Environment Setup

### Create a Virtual Environment
```bash
python3 -m venv .venv
```

### Activate the Environment
```
source .venv/bin/activate  # On Unix/Linux systems
```
```
.env\\Scripts\\activate  # On Windows
```

### Install Dependencies
```
pip install -r requirements.txt
```

## Applications

The presentation layer is available in the `app` folder.<br/>
There are currently two applications under development using jupyter notebooks:

- [Recognition Training](apps/recognition_training.ipynb)
- [Recognition Inference](apps/recognition_inference.ipynb)

### Run the Applications
```
jupyter notebook
```

There are many different ways to explore the notebooks.<br/>
 - Open the browser and navigate to the URL provided by the notebook server.
 - Use the JupyterLab interface if installed.
 - Use one of the existing [vscode extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) if installed.
 - Any other diabolical way you can think of.


## Image Requirements

### Size and Dimension

While the model is designed to work with images of various sizes, it's recommended to use high-resolution images where the features of the subjects are clearly visible.


The provided code automatically resizes images to `299x299` pixels. If you intend to use your own pre-processing steps, make sure to resize the images to this dimension for consistency with the model's architecture.

## Format

The model expects images in RGB format. Make sure your images are RGB and not grayscale or in some other format.

## Preprocessing
The code includes a pre-processing pipeline that resizes the image and converts it into a tensor. If you are providing your own pre-processing steps, ensure it aligns with the model's expected input shape (_Batch Size, 3, 299, 299_).

# License
This project is licensed under the MIT License.