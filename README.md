# Large Language Model Unlearning

Zero Shot Context-Based Object Segmentation using SLIP (SAM+CLIP)

# Goal

The goal of the project is to enhance the capabilities of the SAM (Segment Anything Model [1](#references)) model by incorporating text prompts using CLIP (Contrastive Language-Image Pretraining [2](#references)). This integration, known as SLIP (SAM with CLIP), aims to enable object segmentation without the need for prior training on specific classes or categories. 

# Dataset


# How to Run

Follow the steps below to clone the repository, install the required dependencies, and run the project:

1. **Clone the Repository:**

   ```
     git clone https://github.com/shreya1313/llm-unlearning.git
     cd llm-unlearning
   ```

2. **Install Requirements:**
   
   ```
    pip install -r requirements.txt
   ```
3. **Train Unlearning on Harmful Dataset:**
   
   ```
    python unlearn_harm.py
   ```

4. **Train Classifier:**
   
   ```
    cd classifier
    python train.py
   ```
4. **Evaluate Unlearned Model:**
   
   ```
    cd evaluator
    python evaluate_dataset.py
   ```

# Repository Structure

This repository is organized with a clear structure to help users navigate through different components and functionalities. Below is an overview of the folder structure:

### `classifier/`

- **`data/`**: Contains dataset files for training and testing.
  - `sample_submission.csv`
  - `test.csv`
  - `test_labels.csv`
  - `train.csv`

- **`data_preprocess.py`**: Script for data preprocessing.
- **`download_dataset.py`**: Script for downloading datasets.
- **`eval.py`**: Evaluation script.
- **`inference.py`**: Script for model inference.
- **`model.py`**: Defines the model architecture.
- **`models/`**: Directory to store pre-trained models.
  - `bert-finetuned/`: Example pre-trained model directory.
    - `config.json`
    - `pytorch_model.bin`
    - `special_tokens_map.json`
    - `tokenizer_config.json`
    - `vocab.txt`

- **`scripts/`**: Contains various scripts.
  - `train_2.sh`
  - `train_3.sh`
  - `train.sh`

- **`train.py`**: Training script.
- **`utils.py`**: Utility functions.

### `copyright/`

- Various files related to copyright and licensing.
- **`args.py`**: Script for handling command-line arguments.
- **`config.py`**: Configuration file.
- **`create_dataset.ipynb`**: Jupyter notebook for creating datasets.
- **`extracted_text.jsonl`**: JSON file containing extracted text.
- **`test_unlearned.ipynb`**: Jupyter notebook for testing unlearned models.
- **`train_and_test.ipynb`**: Jupyter notebook for training and testing.
- **`training.py`**: Training script.
- **`training_utils.py`**: Utility functions for training.
- **`unlearned_DatasetDict_small.pkl`**: Pickle file containing unlearned dataset.
- **`unlearn.ipynb`**: Jupyter notebook for unlearning.
- **`utils.py`**: General utility functions.

### `evaluator/`

- Evaluation-related scripts.
  - `evaluate_dataset_opt2.py`
  - `evaluate_dataset.py`
  - `evaluate.py`

### `logs/`

- Log files from various runs.
  - `1.3b_unlearned.out`
  - `baseline_2.out`
  - `baseline_2_test-1.out`
  - ...

### `models/`

- Model directories.
  - `llama-7b_unlearned_ipy/`
    - `adapter_config.json`
    - `adapter_model.bin`
    - `README.md`
  - `opt1.3b_unlearned/`
    - `config.json`
    - `generation_config.json`
    - `pytorch_model.bin`

### `scripts/`

- Various scripts for different tasks.
  - `baseline_2.sh`
  - `baseline_3.sh`
  - `baseline.sh`
  - ...

### `__pycache__/`

- Python bytecode cache.

### `README.md`

- This file, providing an overview of the repository structure.

### `requirements.txt`

- List of dependencies for running the project.

### `run_demo.py`

- Script for running the demonstration.

### `test_baseline_models.ipynb`

- Jupyter notebook for testing baseline models.

### `unlearn_harm.py`

- Script for unlearning harmful information.

### `utils.py`

- General utility functions.

