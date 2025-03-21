## Group 6

# Multimedia Applications: Image Augmentation for Enhanced Machine Learning

## Table of Contents

- [Overview](#overview)
- [Dataset Selection and Justification](#dataset-selection-and-justification)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Project Structure](#project-structure)
- [Data Preprocessing and Augmentation](#data-preprocessing-and-augmentation)
- [Model Architecture and Training](#model-architecture-and-training)
- [Results and Analysis](#results-and-analysis)
- [Usage Instructions](#usage-instructions)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides an in-depth exploration of advanced image augmentation techniques, leveraging the real-world Kaggle dog breed dataset as a case study. The primary objective is to demonstrate the transformative potential of augmentation in enhancing the robustness and reliability of machine learning models. Specifically, the project focuses on how augmentation strategies can address challenges posed by high-resolution images, such as variations in lighting, orientation, and scale.

---

By replicating real-world scenarios through advanced image augmentation techniques, this project highlights the importance of augmentation in enhancing a machine learning model’s capacity to generalize. These techniques help address challenges such as variations in lighting, orientation, and scale, ensuring the model performs effectively across diverse input conditions. The outcomes of this study aim to contribute to the practical applications of machine learning by demonstrating methods to build more reliable and adaptable models, even when working with complex and unpredictable datasets.

---

## Dataset Selection and Justification

- **Dataset Chosen:** [Kaggle Dog Breeds Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

### **Why Dog Breeds?**

1. **Realistic Complexity**  
   The Kaggle Dog Breeds Dataset is a well-curated collection of high-resolution images that mirror real-world complexities encountered in multimedia applications. Each image features diverse background elements, variable lighting conditions, and non-uniform poses of the dogs. This diversity introduces realistic challenges for model training, ensuring that the learning process closely reflects the unpredictability of practical scenarios. By working with these images, we simulate environments that demand robust computer vision solutions.

2. **Fine-Grained Classification**  
   A unique characteristic of this dataset is its focus on fine-grained classification. Unlike broad categorizations (e.g., "animal" vs. "vehicle"), this dataset requires the model to differentiate between visually similar classes, such as various dog breeds with minor distinctions in appearance. Fine-grained classification tasks help strengthen a model's capacity to identify subtle differences, a critical skill for applications like biometric recognition, medical imaging, or product identification in retail systems.

3. **Multimedia Relevance**  
   The dataset is representative of real-world multimedia use cases where image variety is common. It aligns with scenarios such as photo tagging, image-based search engines, and personalized content delivery systems, which rely on accurate identification despite variations in scale, orientation, and quality. By choosing this dataset, we ensure that the resulting techniques are applicable to a wide range of industries beyond academic experimentation.

4. **Data Volume and Quality**  
   The dataset contains a sufficient number of images across multiple classes, providing both breadth and depth for model training. The high quality of images ensures that the data serves as a strong foundation for applying advanced augmentation techniques, ultimately leading to performance improvements.

This dataset not only challenges the model with complex and nuanced inputs but also prepares it for deployment in real-world systems where accuracy and robustness are paramount. It represents a balanced testbed for exploring the impact of augmentation on model performance, making it an ideal choice for this project.

---

## Installation and Environment Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/mufasa101/dogbreed-aug.git
   cd dogbreed-aug
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

### Data Setup

We use the **Stanford Dogs Dataset** from Kaggle. Follow these steps to download and set up the dataset:

1. **Install the Kaggle API:**
   ```bash
   pip install kaggle
   ```
2. **Obtain Kaggle API Credentials:**
   - Log in to your [Kaggle account](https://www.kaggle.com/).
   - Under your profile icon, go to **Account**.
   - Click **Create New API Token**, which downloads `kaggle.json`.
3. **Place the Credentials File:**
   - **Linux/macOS:**
     ```bash
     mkdir -p ~/.kaggle
     mv ~/Downloads/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   - **Windows:**
     1. Create a `.kaggle` folder under `%USERPROFILE%`.
     2. Move `kaggle.json` there.
     3. Adjust file permissions so only you can read the file.
4. **Download the Dataset:**
   ```bash
   kaggle datasets download -d jessicali9530/stanford-dogs-dataset
   ```
5. **Extract the Dataset:**
   ```bash
   mkdir -p data
   unzip stanford-dogs-dataset.zip -d data/
   ```

After this, your `data/` folder will contain the necessary images for training, validation, and testing.

---

## Project Structure

```bash
image_aug/
├── data/                           # Contains downloaded or processed datasets
├── notebooks/                      # Jupyter notebooks for exploration & demos
│   ├── data.ipynb                 # Notebook for exploring & preparing the dataset
│   └── training_experiments.ipynb # Notebook demonstrating model training on original & augmented data
├── src/                            # Source code
│   ├── data_loader.py             # Functions for loading & preprocessing data
│   ├── augmentation.py            # Basic ImageDataGenerator setup
│   ├── custom_augmentation.py     # Albumentations-based augmentations
│   ├── model.py                   # CNN model definition & training routines
│   ├── visualization.py           # Scripts for visualizing original & augmented images
│   └── ui.py                      # Interactive UI components using ipywidgets
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── LICENSE                        # License information
```

---

## Data Preprocessing and Augmentation

### **Preprocessing**

The preprocessing stage is essential in preparing the dataset for efficient model training. It involves several key steps to ensure data quality and compatibility with the machine learning pipeline:

1. **Loading and Splitting the Data**  
   - The `load_images` function is responsible for loading the dataset and dividing it into **training** and **test** sets. This splitting ensures that the model is evaluated on unseen data, promoting robust performance metrics.
   - The function also provides the flexibility to limit the number of folders for quick demonstrations, which is especially useful when testing or debugging the pipeline.

2. **Resizing and Normalizing Images**  
   - All images are resized to a consistent shape of **128×128 pixels**, ensuring uniform input dimensions for the model. This standardization reduces computational overhead and aligns the data with the model's architecture.
   - Normalization scales pixel values to the range **[0, 1]**, stabilizing the model's learning process by reducing variations caused by different image intensity levels.

3. **Error Handling for Data Integrity**  
   - To maintain pipeline stability, the process gracefully skips over corrupt or unreadable files. This ensures that minor data quality issues do not lead to crashes or interruptions during data loading.

---

### **Augmentation Techniques**

To enrich the dataset and simulate real-world variations, this project employs **two distinct approaches** to image augmentation:

1. **Keras/TensorFlow `ImageDataGenerator`**  
   - This built-in TensorFlow tool provides on-the-fly augmentation during model training, efficiently generating modified images directly in memory. Augmentation operations include:
     - **Rotations**: Random rotation within ±30°.
     - **Shifts**: Horizontal and vertical translations to simulate positional variations.
     - **Shear**: Geometric distortions within ±20%.
     - **Zoom**: Random zooming in or out by ±20%.
     - **Horizontal Flips**: Mirroring images to account for symmetry in certain features.  
   - The method is computationally lightweight and integrates seamlessly with the training pipeline, making it ideal for continuous augmentation during model optimization.

2. **Albumentations (Custom Pipeline)**  
   - The Albumentations library is utilized for more advanced and customizable augmentation operations, particularly for visualization purposes. Unlike `ImageDataGenerator`, it enables offline augmentation with a broader range of transformations, such as:
     - **Brightness and Contrast Adjustments**: Modifying lighting conditions to mimic real-world environments.
     - **Random Noise Injection**: Adding noise for better resilience to imperfect input data.
     - **Advanced Geometric Transformations**: Achieving enhanced diversity in the dataset beyond basic shifts and flips.  
   - By visualizing these augmentations, we gain deeper insights into how different transformations influence the dataset and improve the model's ability to generalize.

---
These preprocessing and augmentation techniques work in tandem to ensure the dataset is both high-quality and diverse. Preprocessing provides a clean and standardized input, while augmentation introduces variability to emulate real-world complexities. Together, they lay a solid foundation for training a robust machine learning model capable of performing effectively under various conditions.

---

## **Model Architecture and Training**

### **Architecture**

This project employs a robust **Convolutional Neural Network (CNN)** architecture designed to effectively handle the complex task of high-resolution dog breed classification. The architecture is composed of the following key elements:

1. **Convolution + Pooling Blocks**  
   - Multiple convolutional layers paired with pooling layers allow the network to progressively extract features of increasing complexity.  
   - Early layers focus on detecting basic patterns like edges and textures, while deeper layers identify higher-order features such as shapes and breed-specific details.  
   - Pooling operations reduce the spatial dimensions of feature maps, improving computational efficiency while retaining critical information.

2. **Dropout Layers (50%)**  
   - Dropout is strategically applied to combat overfitting by randomly deactivating 50% of the neurons during training.  
   - This forces the network to generalize better by not overly relying on specific neurons, making the model more robust against unseen data.

3. **Fully Connected Layer**  
   - The final stage of the CNN is a fully connected layer, customized to the total number of dog breeds in the dataset.  
   - The output of this layer is passed through a **softmax activation** function, which converts raw scores into class probabilities, enabling precise breed classification.

---

### **Training Process**

The training pipeline incorporates two key approaches to evaluate the impact of data augmentation:

1. **Original Data Model**  
   - This baseline model is trained on the unaugmented dataset to establish a reference point for performance evaluation.  
   - However, the absence of variability in the input data often leads to overfitting, where the model achieves high accuracy on the training set but performs poorly on validation or test data.

2. **Augmented Data Model**  
   - This model is trained on an augmented dataset created using tools like `ImageDataGenerator` or custom augmentation pipelines from Albumentations.  
   - Augmentation introduces variations in the dataset, such as rotations, shifts, brightness changes, and zooms, which enhance the model's ability to generalize to new, unseen images.  
   - As a result, this approach significantly reduces overfitting and boosts the model's robustness.

Throughout the training process, metrics such as **accuracy** and **loss** are monitored for both the training and validation sets across multiple epochs. This enables a detailed comparison of how the models learn and adapt.

---

## **Results and Analysis**

1. **Performance Curves**  
   - The training and validation performance is visualized using accuracy and loss curves.  
   - Comparisons reveal that the augmented model generally narrows the gap between training and validation performance, highlighting improved generalization.

2. **Confusion Matrix**  
   - A confusion matrix provides a detailed analysis of the model’s predictions for each breed.  
   - It identifies breeds that are frequently misclassified, offering insights into specific areas where augmentation has enhanced performance or where further adjustments may be necessary.

3. **Visual Comparisons**  
   - Side-by-side visualizations of original and augmented images showcase the diversity introduced through augmentation techniques.  
   - Transformations such as rotation, shifting, and zooming enrich the dataset, giving the model a more representative view of real-world image variations.

---

### **Key Takeaway**

This project demonstrates the power of data augmentation in addressing overfitting and enhancing a model’s ability to handle real-world conditions. By integrating diverse augmentation techniques, the CNN not only achieves higher accuracy but also becomes more adaptable and reliable in high-resolution, fine-grained classification tasks such as dog breed identification.

---

## Usage Instruction

1. **Data Acquisition:**  
   Follow the [Data Setup](#data-setup) instructions to download and organize the Stanford Dogs Dataset.

Below is an example of how you might **introduce and document** a separate notebook called **`data.ipynb`** in your project. This notebook can serve as a dedicated space for **data exploration, cleaning, and preprocessing** steps, keeping your workflow organized and easy to follow.

---

## **`data.ipynb`** – Data Exploration and Preprocessing

This notebook is designed to **explore** and **prepare** the dataset before training any models. It allows you to visually inspect images, check label distributions, and apply basic preprocessing or cleaning steps.

### **Notebook Overview**

1. **Dataset Inspection:**

   - Examine folder structure and verify that images are organized as expected.
   - Print sample file paths, display random images, and confirm they’re properly labeled.
   - Check for potential issues like missing or corrupt files.

2. **EDA (Exploratory Data Analysis):**

   - Plot the distribution of classes (e.g., how many images per breed).
   - Identify imbalances or underrepresented classes.
   - Possibly visualize basic statistics (image dimensions, aspect ratios, etc.).

3. **Preprocessing:**

   - Resize images to a consistent shape (e.g., 128×128).
   - Normalize pixel values to a 0–1 range or other standardization.
   - (Optional) Crop or remove unnecessary margins if needed.

4. **Splitting or Reorganizing:**

   - If your dataset isn’t already split into training and testing sets, do so here (using `train_test_split` or a custom approach).
   - Move or copy files into `train/` and `val/` directories, or handle them with code in `src/data_loader.py`.

5. **Documentation of Findings:**
   - Record any peculiarities discovered, such as corrupt images or mislabeled samples.
   - Suggest strategies for dealing with heavily imbalanced classes or low-quality images.

### **Why Have a Separate `data.ipynb`?**

- **Cleaner Workflow:** By isolating data exploration and cleaning tasks in one notebook, you keep your main training notebook (`training_experiments.ipynb`) focused on model-related steps.
- **Reproducibility:** Anyone can open `data.ipynb` to understand how you prepared the dataset before training.
- **Debugging:** If there are discrepancies in the data (e.g., class distribution doesn’t match expectations), you can revisit this notebook to pinpoint where the process might have gone awry.

### **Usage Instructions**

1. **Open `data.ipynb`:**
   ```bash
   jupyter notebook notebooks/data.ipynb
   ```
2. **Run Cells in Order:**  
   Start from the top to load images, examine distributions, and perform any required cleaning or splitting.
3. **Confirm Outputs:**  
   Check that the final distribution of images aligns with your project’s needs. If you’re limiting folders or images for demonstration, verify that the code handles these cases gracefully.

### **Next Steps**

Once you’ve verified and preprocessed the data in `data.ipynb`, you can move on to:

- **`training_experiments.ipynb`:** For model training on original vs. augmented data.
- **`ui.py` / Interactive Visualization:** For real-time augmentation demos and image comparisons.

---

### **Training & Experiments:**

Open and run the notebook in `notebooks/training_experiments.ipynb` to:

- Train the model on **original data**.
- Train the model on **augmented data**.
- Compare the two training runs with performance plots and confusion matrices.

### **Visualization:**

- Use `src/visualization.py` or the interactive UI (`src/ui.py`) to visualize original vs. augmented images.
- Experiment with `custom_augmentation.py` (Albumentations) for a more advanced augmentation pipeline.

### **Interactive UI (Optional):**

- Run `%run -i src/ui.py` in a Jupyter cell to load the dataset, visualize images, and adjust augmentation parameters on the fly.

---

## Contributing

We welcome feedback and contributions! Whether you’d like to refine the augmentation pipeline, experiment with new model architectures, or improve documentation, feel free to open an issue or submit a pull request. Please ensure your changes are well-documented and tested.

---

## License

This project is distributed under the [MIT License](LICENSE). You’re free to use, modify, and distribute this code, provided you include proper attribution.

---

**Group 6** thanks you for exploring **Image Augmentation for Enhanced Machine Learning**. If you encounter any issues or have suggestions for improvement, please open an issue or reach out via our discussion boards. We look forward to your contributions!
