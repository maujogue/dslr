## DSLR Project

Welcome to the DSLR project!  
This project will guide you through:

- Loading and exploring datasets
- Visualizing data in multiple ways
- Cleaning and preprocessing your data
- Training a logistic regression model to solve a classification problem

---

### 🚀 Installation

To set up the project and install all dependencies, run:

```zsh
$ make install
```

```
Dependencies installed successfully!
To activate the virtual environment, run:
source config/.venv_dslr/bin/activate
```

Now, activate the env :

```zsh
$ source config/.venv_dslr/bin/activate
```

```
(.venv_dslr) $
```

### Running the program

Available programs are located in the `src/` folder. You can run them with python3:

```zsh
python3 src/describe.py
```

or print the help for more information on the program

```zsh
python3 src/describe.py --help
```

### Programs

#### Data Analysis

- `describe.py` [file] [--advanced]
  - **Arguments:**
    - `file` (optional): Path to the CSV file to describe (default: `datasets/dataset_train.csv`)
  - **Options:**
    - `-h`, `--help`: Show help message and exit
    - `-a`, `--advanced`: Include advanced statistics (missing, unique, iqr)
  - **Purpose:** Provides comprehensive statistical description of numerical columns in the dataset

#### Data Visualization

- `histogram.py` [file]

  - **Arguments:**
    - `file` (optional): Path to the CSV file to visualize (default: `datasets/dataset_train.csv`)
  - **Purpose:** Creates histograms for each numerical feature, grouped by Hogwarts House, showing the distribution of grades across different houses

- `pair_plot.py` [file]

  - **Arguments:**
    - `file` (optional): Path to the CSV file to visualize (default: `datasets/dataset_train.csv`)
  - **Purpose:** Generates a pair plot matrix showing relationships between different features, with points colored by Hogwarts House

- `scatter_plot.py` [file]
  - **Arguments:**
    - `file` (optional): Path to the CSV file to visualize (default: `datasets/dataset_train.csv`)
  - **Purpose:** Creates scatter plots to analyze correlations between specific features (e.g., Defense Against the Dark Arts vs Astronomy)

#### Machine Learning

- `logreg_train.py` [file]

  - **Arguments:**
    - `file` (optional): Path to the CSV file to train on (default: `datasets/dataset_train.csv`)
  - **Purpose:** Trains a one-versus-all logistic regression model to classify students into Hogwarts Houses based on their academic performance
  - **Features used:** Potions, Charms, Transfiguration, Astronomy, Divination, Ancient Runes, Muggle Studies
  - **Output:** Saves trained weights to `weights.txt`

- `logreg_predict.py` [file]
  - **Arguments:**
    - `file` (optional): Path to the CSV file to predict on (default: `datasets/dataset_test.csv`)
  - **Purpose:** Uses the trained model to predict Hogwarts House assignments for new students
  - **Output:** Saves predictions to `houses.csv` and displays results in terminal

### Example Usage

```zsh
# Visualize data distributions
python3 src/histogram.py datasets/dataset_train.csv

# Analyze feature relationships
python3 src/pair_plot.py datasets/dataset_train.csv

# Check specific correlations
python3 src/scatter_plot.py datasets/dataset_train.csv

# Train the model
python3 src/logreg_train.py datasets/dataset_train.csv

# Make predictions on test data
python3 src/logreg_predict.py datasets/dataset_test.csv
```

### Model Details

The logistic regression model implements **one-versus-all classification**:

- Creates 4 binary classifiers (one for each house)
- Each classifier predicts whether a student belongs to that specific house or not
- Features are standardized using z-score normalization
- Uses sigmoid activation and binary cross-entropy loss
- Trains for up to 10,000 iterations with early stopping

### Output Files

- `weights.txt`: Contains trained model weights for each house
- `houses.csv`: Contains predictions for test dataset in the format required for evaluation
