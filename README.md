# Text Classifier

## Overview

This project implements a text classification model using the Hugging Face Transformers library. The model is based on the pre-trained "badalsahani/text-classification-multi" model available on Hugging Face's model hub.

The code takes user input text, preprocesses it using the model's tokenizer, performs a forward pass through the model, and outputs the predicted class along with class probabilities.

## Requirements

- Python 3.x
- Hugging Face Transformers library
- Torch library

## Installation

1. Install the required dependencies:

   ```bash
   pip install transformers torch

2. Download the code from the repository:

   ```bash
   git clone [(https://github.com/NEMERO21/Text-Classifier.git)https://github.com/NEMERO21/Text-Classifier.git]
   cd Text-Classifier

3. Set up your Hugging Face API token:
   - Sign up or log in to your Hugging Face account.
   - Retrieve your API token from Hugging Face.
   - Set the token using the following command:
     ```bash
     export HF_HOME=~/.huggingface
     
4. Run the script:
   ```bash
   python text_classifier.py

## Usage 

1. When prompted, enter the text you want to classify.
   
2. The script will preprocess the text, pass it through the model, and output the predicted class along with class probabilities.

## Model Information
 - Model Name: badalsahani/text-classification-multi
 - Classes:
     - Biology
     - Business Studies
     - Chemistry
     - Computer Science
     - Economics
     - English Literature
     - Geography
     - History
     - Mathematics
     - Physics
     - Political Science
     - Psychology
     - Sociology
  
## License 

This project is licensed under the MIT License.

## Acknowledgements
- The Hugging Face community for providing pre-trained models and the Transformers library.
