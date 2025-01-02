# Load required libraries
install.packages("BiocManager")
BiocManager::install("EBImage")
install.packages("randomForest")
install.packages("ggplot2")
install.packages("xml2")
install.packages("magrittr")
install.packages("tidyverse")
install.packages("keras")
install.packages("reticulate")
install.packages("tensorflow")

library(EBImage)
library(randomForest)
library(ggplot2)

library(xml2)

library(magrittr)

library(tidyverse)

library(keras)

library(reticulate)

library(tensorflow)
install_tensorflow()


preprocess_image <- function(image_path, target_size = c(256, 256)) {
  # Determine file type and load image accordingly
  if (grepl("\\.jpg$|\\.jpeg$", image_path, ignore.case = TRUE)) {
    img <- readJPEG(image_path)
  } else if (grepl("\\.png$", image_path, ignore.case = TRUE)) {
    img <- readPNG(image_path)
  }
  
  # Display original image
  display(img, method = "raster", title = "Original Image", all = TRUE)
  
  # Convert to grayscale
  img_gray <- channel(img, 'gray')
  display(img_gray, method = "raster", title = "Grayscale Image", all = TRUE)
  
  # Resize image
  img_resized <- resize(img_gray, target_size[1], target_size[2])
  display(img_resized, method = "raster", title = "Resized Image", all = TRUE)
  
  # Normalize the image
  img_normalized <- img_resized / max(img_resized)
  display(img_normalized, method = "raster", title = "Normalized Image", all = TRUE)
  
  # Augmentation (flip and rotate)
  img_flipped <- flip(img_normalized, 'x')  # Horizontal flip
  display(img_flipped, method = "raster", title = "Flipped Image", all = TRUE)
  
  img_rotated <- rotate(img_normalized, angle = 15)  # Rotate by 15 degrees
  display(img_rotated, method = "raster", title = "Rotated Image", all = TRUE)
  
  preprocessed_list <- list(original = img_normalized, flipped = img_flipped, rotated = img_rotated)
  print(paste("Processed", length(preprocessed_list), "images from:", basename(image_path)))
  
  return(preprocessed_list)
}

# Apply feature extraction to only one version (original) in the list
extract_all_features <- function(preprocessed_images) {
  original_image <- preprocessed_images$original
  features <- extract_features(original_image)
  return(features)
}

# Function to apply PCA
apply_pca <- function(features, n_components = 50) {
  pca <- prcomp(features, center = TRUE, scale. = TRUE)
  pca_features <- pca$x[, 1:n_components]
  return(pca_features)
}

# Function to extract object names from XML
extract_object_names <- function(xml_file_path) {
  # Read the XML file
  xml_data <- read_xml(xml_file_path)
  
  # Extract object names
  object_names <- xml_data %>%
    xml_find_all(".//object/name") %>%
    xml_text()
  
  # Assuming one object name per image, take the first
  return(object_names[1])
}


# obtaining labels
train_annotations_path <- "/Users/abhishek/Downloads/dataset/annotations/train"
test_annotations_path <- "/Users/abhishek/Downloads/dataset/annotations/test"

# List XML files in train and test annotations folders
train_xml_files <- list.files(train_annotations_path, full.names = TRUE, pattern = "*.xml")
test_xml_files <- list.files(test_annotations_path, full.names = TRUE, pattern = "*.xml")

# Extract object names as labels from train XML files
train_labels <- sapply(train_xml_files, extract_object_names)
train_labels <- as.factor(train_labels)

# Extract object names as labels from test XML files
test_labels <- sapply(test_xml_files, extract_object_names)
test_labels <- as.factor(test_labels)

# Print Training and Testing Labels
print("Train Labels:")
print(train_labels)

print("Test Labels:")
print(test_labels)


train_files <- list.files("/Users/abhishek/Downloads/dataset/images/train", full.names = TRUE)
print(paste("Number of training files:", length(train_files)))
train_processed <- lapply(train_files, preprocess_image)
train_features <- do.call(rbind, lapply(train_processed, extract_all_features))
print(paste("Number of training features:", nrow(train_features)))
print(paste("Number of training labels:", length(train_labels)))

# Print the shape of the training features
shape <- dim(train_features)
cat("Number of rows (images):", shape[1], "\n")
cat("Number of columns (features):", shape[2], "\n")


# Load testing images, preprocess and extract features
test_images_path <- "/Users/abhishek/Downloads/dataset/images/test"
test_files <- list.files(test_images_path, full.names = TRUE)
print(paste("Number of test files:", length(test_files)))
test_processed <- lapply(test_files, preprocess_image)
test_features <- do.call(rbind, lapply(test_processed, extract_all_features))
print(paste("Number of testing features:", nrow(test_features)))
print(paste("Number of testing labels:", length(test_labels)))

train_features_pca <- apply_pca(train_features, n_components = 50)
# Apply PCA for dimensionality reduction on test data
test_features_pca <- predict(prcomp(train_features, center = TRUE, scale. = TRUE), newdata = test_features)
test_features_pca <- test_features_pca[, 1:50]

# Ensure number of features matches number of labels
print(paste("Number of training features:", nrow(train_features)))
print(paste("Number of training labels:", length(train_labels)))

# Verify the unique labels
print("Unique Training Labels:")
print(unique(train_labels))

# Correct any mismatches if necessary
if (nrow(train_features) != length(train_labels)) {
  stop("Number of features does not match number of labels.")
}

# Assuming 'train_labels' is your target variable vector
class_distribution <- table(train_labels)
print(class_distribution)

# Proportion of each class
class_proportions <- prop.table(class_distribution)
print(class_proportions)

class_proportions <- prop.table(class_distribution)
print(class_proportions)

# Using base R to plot
barplot(table(train_labels), main = "Class Distribution", xlab = "Classes", ylab = "Frequency", col = "blue")

# Using ggplot2 for a more advanced visualization
if(!require(ggplot2)) install.packages("ggplot2")
library(ggplot2)
ggplot(data = data.frame(Class = train_labels), aes(x = Class)) +
  geom_bar(fill = "blue") +
  labs(title = "Class Distribution", x = "Classes", y = "Count")


library(randomForest)
# Train the Random Forest model
rf_model <- randomForest(train_features_pca, train_labels, ntree = 200)

# Tuning Random Forest hyperparameters
set.seed(42)  # For reproducibility
rf_model <- randomForest(
  x = train_features_pca, 
  y = train_labels, 
  ntree = 300,                      # Increased number of trees
  mtry = sqrt(ncol(train_features_pca)), # Default suggestion, can be tuned
  maxnodes = NULL,                  # Unlimited number of nodes
  nodesize = 1,                     # Minimum size of terminal nodes
  importance = TRUE,                # Calculate variable importance
  na.action = na.omit               # Handling missing values by omitting
)

# Print model summary
print(rf_model)


predictions <- predict(rf_model, test_features_pca)
accuracy <- mean(predictions == test_labels)
cat("Random Forest Accuracy: ", accuracy * 100, "%\n")


# Checking variable importance
importance(rf_model)
varImpPlot(rf_model)


library(caret)

# Assume 'predictions' and 'test_labels' are already defined
conf_matrix <- confusionMatrix(data = predictions, reference = test_labels)

# Print the confusion matrix
print(conf_matrix$table)

# Simple plot (provided by the caret package)
plot(conf_matrix$table, main = "Confusion Matrix")

library(ggplot2)

# Compute the confusion matrix as a table
conf_mat <- table(Predicted = predictions, Actual = test_labels)

# Convert the table to a dataframe for ggplot
conf_mat_df <- as.data.frame(as.table(conf_mat))

# Plot using ggplot2
ggplot(data = conf_mat_df, aes(x = Predicted, y = Actual, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1.5, color = "white") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(title = "Confusion Matrix", x = "Predicted Label", y = "Actual Label") +
  theme_minimal()

print(predictions)

# Convert predictions to a data frame
predictions_df <- data.frame(Predictions = predictions)

# Plot distribution of predictions
ggplot(predictions_df, aes(x = Predictions)) +
  geom_bar(fill = "blue") +
  labs(title = "Distribution of Predictions", x = "Classes", y = "Count") +
  theme_minimal()


# Convert labels to data frames for ggplot
train_labels_df <- data.frame(Label = factor(train_labels))
test_labels_df <- data.frame(Label = factor(test_labels))

# Plot for Training Labels
ggplot(train_labels_df, aes(x = Label)) +
  geom_bar(fill = "pink") +
  labs(title = "Distribution of Training Labels", x = "Labels", y = "Count") +
  theme_minimal()

combined_labels <- data.frame(
  Label = c(as.character(train_labels), as.character(test_labels)),
  Set = factor(c(rep("Training", length(train_labels)), rep("Test", length(test_labels))))
)
# Plot for Test Labels
ggplot(test_labels_df, aes(x = Label)) +
  geom_bar(fill = "orange") +
  labs(title = "Distribution of Test Labels", x = "Labels", y = "Count") +
  theme_minimal()


# Create a density plot
ggplot(combined_labels, aes(x = Label, fill = Set, color = Set)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Labels", x = "Labels", y = "Density")