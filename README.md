AI Training Project (Binary Classification)
This project implements a binary classification AI using a self-developed training algorithm, including a custom gradient descent algorithm.

Training Code (training.py)
Contains the model and kernel class definitions.
The “apply” function in training.py has a variable called thres (threshold).
To maximize accuracy, set the thres value based on trained models.
The scale of the model can be changed in the "__init__" function in class "model". Details can be found in the comments in the code

Using Pre-Trained Models
The repository includes three pre-trained models in .txt format. Their filenames follow this structure:
"accuracy threshold value.txt"
Example:
0.95 threshold 0.85.txt  →  Accuracy: 95%, Threshold: 0.85

How to Use a Pre-Trained Model:
Rename the desired model file to AIdata2.txt.
Run training.py.
This will load and apply the selected model.
Continue running training.py will train and optimize the model.
