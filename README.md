This project studies knowledge distillation for automated essay scoring with ordinal labels. 

The data is gathered from Kaggle:

https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2

The aim is to explore how complex machine learning models can transfer their behaviour to simpler and more interpretable models.


This project uses a `src/` layout. To make imports work from anywhere (notebooks, scripts), install the project in editable mode.

Setup (PowerShell, from repo root):

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

After this, you can import project code from anywhere:

from src.models.teacher import TeacherTrainer, TeacherConfig
from src.utils.metrics import quadratic_weighted_kappa

Also remember for spacy:

python -m spacy download en_core_web_sm
