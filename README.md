# LLM ChatBot

This is a simple and basic implementation of a ChatBot using Transformer LLM and a Cornell Movies Dataset

## Installation

 - Create a virtual environment in this project directory:
    ```bash
    python -m venv .
    ```

 - Activate it:
    ```bash
    Scripts\activate #Windows
    ```
    ```bash
    source ./bin/activate #Linux
    ```

 - Install all required packages:
    ```bash
    pip install -r requirements.txt
    ```

 - To deactivate it:
    ```bash
    Scripts\deactivate #Windows
    ```
    ```bash
    source ./bin/deactivate #Linux
    ```

## Usage

### For training the model:
```bash
python src/train.py [optional -p or -v]
```
|Flag|Meaning|
|----|-------|
|`-p`|load existing preprocessed data and vectorize it|
|`-v`|load existing vectorized data|
|None|load original raw data, preprocess and vectorize it|

The model will load the data, preprocess and vectorize it, and train the Transformer model on it. The model is able to detect and use CUDA GPU.

### For running and chatting with model:
```bash
python src/run.py
```
You will receive the following instruction:
```
Chat with the model! Type 'exit' to exit:
You:
```
Type something and experiment with the model. Enjoy!

**Note: You must train the model before running it!**

## Notebooks
Navigate `notebooks/` to run them for research purposes.

**Note: Please read the [`README`](notebooks/README.md) before running any notebook!**