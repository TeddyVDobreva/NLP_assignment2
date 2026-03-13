The repository has the following structure:
.
├── data
│   ├── fast.txt
│   ├── test.csv
│   └── train.csv
├── main.py
├── plots
│   ├── confusion_matrix_CNN_testing.png
│   ├── confusion_matrix_CNN_validation.png
│   ├── confusion_matrix_LSTM_testing.png
│   ├── confusion_matrix_LSTM_validation.png
│   ├── heatmap_lr_embed_dim.png
│   ├── learning_curves__f1.png
│   ├── learning_curves_loss.png
│   └── lengths_distribution.png
├── plots copy
│   ├── heatmap_lr_embed_dim.png
│   └── lengths_distribution.png
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── __pycache__
│   │   ├── data_handler.cpython-311.pyc
│   │   ├── functions_models.cpython-311.pyc
│   │   ├── hyperparameter_evaluation.cpython-311.pyc
│   │   └── models.cpython-311.pyc
│   ├── ablation.py
│   ├── data_handler.py
│   ├── evaluation.py
│   ├── hyperparameter_evaluation.py
│   └── models.py
└── uv.lock

You can run the code by following the instructions enclosed here.

First, clone the repository by running:
- for SSH
```bash
git clone git@github.com:TeddyVDobreva/NLP_assignment2.git
```
- for HTTPS
```bash
git clone https://github.com/TeddyVDobreva/NLP_assignment2.git
```

After cloning the repository, you want to activate the uv environement.

```bash
uv sync
```
```bash
uv source .venv/bin/activate
```

After this, you are ready run the code in the terminal:

python main.py