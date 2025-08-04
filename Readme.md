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

Available programs are located in the ```src/``` folder. You can run them with python3:
```zsh
python3 src/describe.py
```

or print the help for more information on the program
```zsh
python3 src/describe.py --help
```

### Programs

- `describe.py` [file] [--advanced]
  - **Arguments:**
    - `file` (optional): Path to the CSV file to describe (default: `datasets/dataset_train.csv`)
  - **Options:**
    - `-h`, `--help`: Show help message and exit
    - `-a`, `--advanced`: Include advanced statistics (missing, unique, iqr)


