
# Medical Image Classification with Chest X-Ray Images using CNN 
==============================
In this project, I developed a Convolutional Neural Network (CNN) model for classifying chest x-ray images. The aim of the project was to build a model that could accurately detect the presence of lung diseases, such as pneumonia, in chest x-ray images.

## Tools and Libraries

This project uses the following tools and libraries:

- **Python**
- **NumPy**
- **OpenCV**
- **TensorFlow/Keras**
- **scikit-learn**
- **Flask**
- **Matplotlib**
- **imbalanced-learn**

Make sure to install and set up these tools and libraries before running the project.
### Steps

1. **Read the Images:** First, you need to read the images you'll be working with. You can use various libraries like OpenCV, PIL, or scikit-image to read the images into your Python code.

2. **Pre-process and Rescale the Images:** Pre-process the images to prepare them for training and testing. This may involve resizing, normalization, data augmentation, etc.

3. **Exploratory Data Analysis (EDA)**

4. **Convolutional Neural Networks (CNNs):** Implement your Convolutional Neural Network architecture using Keras and TensorFlow. Define the layers, architecture, and configurations of your CNN.

5. **Training the Model:** Train your model on a training dataset. Specify the number of epochs, batch size, and optimization techniques you plan to use. Include code snippets or links to relevant code files.

6. **Model Evaluation:** Evaluate the model's performance on a test set of images. Report evaluation metrics such as accuracy, loss, precision, recall, and F1-score.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
