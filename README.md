# Welding-Quality-Detection-System
This project simulates an Intelligent Welding Quality Assurance System. In real-world manufacturing, welding defects like Burn Through, Incomplete Fusion, and Porosity can compromise structural integrity. This system uses Machine Learning to:  Predict Weld Strength: Estimate tensile strength based on process parameters 

# Welding Quality Detection System

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Generate Synthetic Data**:
    ```bash
    python src/data_gen.py
    ```
    This creates `data/welding_data.csv` and `data/images/*.png`.

2.  **Train Tabular Models**:
    ```bash
    python src/tabular_model.py
    ```
    Trains Random Forest models for Strength, Pass/Fail, and Defect classification.

3.  **Train Computer Vision Model**:
    ```bash
    python src/cv_model.py
    ```
    Trains a simple CNN on the generated weld images.

4.  **Run Main System**:
    ```bash
    python main.py
    ```
    Runs a test case through the full pipeline.
