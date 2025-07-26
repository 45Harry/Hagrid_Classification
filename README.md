# Hagrid Classification Project

A deep learning project for gesture classification using the Hagrid dataset. This project implements a Convolutional Neural Network (CNN) to classify hand gestures from images.

## 📋 Project Overview

This project uses TensorFlow/Keras to build and train a CNN model for classifying hand gestures. The model is based on MobileNet architecture and is designed to work with the Hagrid dataset, which contains various hand gesture images.

### Key Features

- **Transfer Learning**: Uses pre-trained MobileNet as the base model
- **Data Augmentation**: Implements data augmentation techniques for better model generalization
- **Model Optimization**: Includes callbacks for learning rate reduction and early stopping
- **Comprehensive Evaluation**: Provides training history visualization and prediction capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- Sufficient disk space for the dataset (several GB)

### Installation

1. **Clone the repository**
   ```bash
   git clone 'https://github.com/45Harry/Hagrid_Classification'
   cd Hagrid_Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - The project expects the Hagrid dataset to be in the `datasets/` directory
   - Dataset structure should be:
     ```
     datasets/
     └── hagrid-classification-small/
         ├── class1/
         ├── class2/
         └── ...
     ```

## 📁 Project Structure

```
Hagrid_Classification/
├── datasets/                          # Dataset directory
│   └── hagrid-classification-small/   # Processed dataset
├── Hagrid_Classification_2.ipynb      # Main Jupyter notebook
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## 🔧 Usage

### Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   - Open `Hagrid_Classification_2.ipynb`

3. **Follow the notebook sections**:
   - Data preprocessing and augmentation
   - Model creation and compilation
   - Training the model
   - Evaluation and predictions

### Key Components

#### Data Preprocessing
- Dataset loading using `tf.keras.preprocessing.image_dataset_from_directory`
- Data augmentation with rotation, zoom, and flip operations
- Train/validation/test split (80/10/10)

#### Model Architecture
- **Base Model**: MobileNet (pre-trained, frozen)
- **Additional Layers**:
  - Global Average Pooling
  - Dropout (0.2)
  - Dense layer (output classes)
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy

#### Training Configuration
- **Batch Size**: 32
- **Epochs**: 20
- **Callbacks**:
  - ReduceLROnPlateau
  - EarlyStopping

## 📊 Model Performance

The model achieves classification accuracy on the Hagrid gesture dataset. Performance metrics include:
- Training accuracy
- Validation accuracy
- Loss curves
- Confusion matrix

## 🛠️ Customization

### Modifying the Model
- Change the base model in the `base_model` section
- Adjust the number of classes in the final dense layer
- Modify data augmentation parameters

### Dataset Configuration
- Update the dataset path in the notebook
- Modify the train/validation/test split ratios
- Adjust image size and batch size as needed

## 📝 Dependencies

See `requirements.txt` for the complete list of dependencies. Key packages include:
- TensorFlow 2.x
- NumPy
- Matplotlib
- Jupyter

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hagrid dataset creators
- TensorFlow/Keras community
- MobileNet architecture developers

## 📞 Support

For questions or issues, please open an issue in the repository or contact the maintainers.

---

**Note**: This project is designed for educational and research purposes. Make sure to comply with the dataset's terms of use and licensing requirements. 