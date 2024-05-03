import numpy as np
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump

# Open the sentinel image
sentinel_dataset = gdal.Open(r"G:\America_CTD\America_CTD\AOI\Sentinel2_AOI2021.tif", gdal.GA_ReadOnly)
if sentinel_dataset is None:
    print("Failed to open the sentinel image.")
    exit(1)

from sklearn.impute import SimpleImputer

# Open the crop type image
crop_type_dataset = gdal.Open(r"G:\America_CTD\America_CTD\AOI\CropMask_AOI2021.tif", gdal.GA_ReadOnly)
if crop_type_dataset is None:
    print("Failed to open the crop type image.")
    exit(1)

width = crop_type_dataset.RasterXSize
height = crop_type_dataset.RasterYSize

# Get image dimensions
width = sentinel_dataset.RasterXSize
height = sentinel_dataset.RasterYSize
num_bands_sentinel = sentinel_dataset.RasterCount
num_bands_crop_type = crop_type_dataset.RasterCount

# Read the sentinel image bands
sentinel_image = []
for band_num in range(1, num_bands_sentinel + 1):
    band = sentinel_dataset.GetRasterBand(band_num)
    band_data = band.ReadAsArray()
    sentinel_image.append(band_data)

# Read the crop type image bands
crop_type_image = []
for band_num in range(1, num_bands_crop_type + 1):
    band = crop_type_dataset.GetRasterBand(band_num)
    band_data = band.ReadAsArray()
    crop_type_image.append(band_data)

# Convert the lists to numpy arrays
sentinel_image = np.stack(sentinel_image, axis=-1)
crop_type_image = np.stack(crop_type_image, axis=-1)

# Close the datasets
sentinel_dataset = None
crop_type_dataset = None

# Reshape the images
sentinel_image = sentinel_image.reshape(-1, num_bands_sentinel)
crop_type_image = crop_type_image.flatten()

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(sentinel_image, crop_type_image, test_size=0.2, random_state=42)

# Define and train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=40)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:\n", conf_matrix)

# Save the trained model
model_path = r"G:\America_CTD\CTD\America_CTD_10.joblib"
dump(rf_classifier, model_path)

print("Model saved successfully.")

# =============================================================================
# =============================================================================
# # Prediction
# =============================================================================
# =============================================================================

import numpy as np
from osgeo import gdal
from joblib import load
from sklearn.impute import SimpleImputer

# Load the new image for prediction
new_image_path = r"E:\Canada\New_AOI\Validation\Composite_Image.tif"
new_image_dataset = gdal.Open(new_image_path, gdal.GA_ReadOnly)
if new_image_dataset is None:
    print("Failed to open the new image.")
    exit(1)

# Get image dimensions
width = new_image_dataset.RasterXSize
height = new_image_dataset.RasterYSize
num_bands = new_image_dataset.RasterCount

# Read the new image bands
new_image = []
for band_num in range(1, num_bands + 1):
    band = new_image_dataset.GetRasterBand(band_num)
    band_data = band.ReadAsArray()
    new_image.append(band_data)

# Convert the list to numpy array
new_image = np.stack(new_image, axis=-1)

# Reshape the image
new_image_reshaped = new_image.reshape(-1, num_bands)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
new_image_reshaped_imputed = imputer.fit_transform(new_image_reshaped)

# Load the trained model
model_path = r"E:\Canada\New_AOI\Model\Random_Forest50.joblib"
loaded_rf_classifier = load(model_path)

# Make predictions on the new image
predicted_classes = loaded_rf_classifier.predict(new_image_reshaped_imputed)

# Reshape the predictions to the original image dimensions
predicted_classes_image = predicted_classes.reshape((height, width))

# Save the prediction image
output_path = r"E:\Canada\New_AOI\Validation\Pred_New_Accurate.tif"
driver = gdal.GetDriverByName("GTiff")
output_dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Float32)
output_dataset.GetRasterBand(1).WriteArray(predicted_classes_image)

# Assign coordinate information from the new image
output_dataset.SetGeoTransform(new_image_dataset.GetGeoTransform())

# Assign projection from input image
output_dataset.SetProjection(new_image_dataset.GetProjection())

output_dataset.FlushCache()
output_dataset = None

print("Prediction image saved successfully.")