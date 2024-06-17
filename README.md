# Crop Type Classification Using Random Forest

This project classifies crop types from Sentinel-2 imagery using a Random Forest classifier. The workflow includes training the model with Sentinel-2 imagery and crop type data, evaluating the model, and applying the model to predict crop types in a new area.

## Requirements

- Python 3.x
- NumPy
- GDAL
- scikit-learn
- joblib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/crop-type-classification.git
    cd crop-type-classification
    ```

2. Install the required Python packages:

    ```bash
    pip install numpy gdal scikit-learn joblib
    ```

## Usage

### Training the Model

1. Ensure you have the required input files:
   - Sentinel-2 image (e.g., `Sentinel2_AOI2021.tif`)
   - Crop type image (e.g., `CropMask_AOI2021.tif`)

2. Modify the script paths in `train_model.py` to point to your input files.

3. Run the script to train the model:

    ```python
    python train_model.py
    ```

4. The script will output evaluation metrics and save the trained model as `America_CTD_10.joblib`.

### Making Predictions

1. Ensure you have the trained model and the new image for prediction:
   - New composite image (e.g., `Composite_Image.tif`)
   - Trained model (e.g., `Random_Forest50.joblib`)

2. Modify the script paths in `predict.py` to point to your input files and model.

3. Run the script to make predictions:

    ```python
    python predict.py
    ```

4. The script will save the predicted classes as `Pred_New_Accurate.tif`.

## Functions

### `train_model.py`

#### `calculate_metrics(y_test, y_pred)`

Calculates and prints evaluation metrics for the model.

- **Parameters:**
  - `y_test` (array): True labels of the test set.
  - `y_pred` (array): Predicted labels of the test set.
- **Returns:** None

#### `save_model(model, model_path)`

Saves the trained model to a specified path.

- **Parameters:**
  - `model` (sklearn model): Trained model.
  - `model_path` (str): Path to save the model.
- **Returns:** None

### `predict.py`

#### `load_image(image_path)`

Loads a raster image and returns it as a numpy array.

- **Parameters:**
  - `image_path` (str): Path to the image file.
- **Returns:** numpy.ndarray: Image data.

#### `impute_missing_values(image_reshaped)`

Handles missing values in the image data using mean imputation.

- **Parameters:**
  - `image_reshaped` (numpy.ndarray): Reshaped image data.
- **Returns:** numpy.ndarray: Image data with imputed values.

#### `save_prediction_image(predicted_classes, output_path, reference_image_path)`

Saves the prediction results as a raster image.

- **Parameters:**
  - `predicted_classes` (numpy.ndarray): Predicted classes data.
  - `output_path` (str): Path to save the prediction image.
  - `reference_image_path` (str): Path to the reference image for geotransform and projection information.
- **Returns:** None


