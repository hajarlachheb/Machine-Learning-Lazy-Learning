### Running script for the first time

# Open project’s root folder in terminal:
 ```bash
cd <root_folder_of_project>/ 
```

# Create virtual environment: 
 ```bash
python3 -m venv venv/ 
```

# Open virtual environment:
 ```bash
source venv/bin/activate 
```

# Install required dependencies: 
```bash
pip install -r requirements.txt 
```

# Close virtual environment: 
```bash
deactivate
```

## Execute scripts
# Open folder in terminal: 
```bash
cd <root_folder_of_project>/
 ```

# Open the virtual environment:
```bash
source venv/bin/activate
```

# Run all experiments for a given data set
In this main file the dataset for which to run experiments is entered by the user through the command window.
It will execute all the possible combinations for the k-NN algorithm for the given data set and will store the results
in two text files (.txt) for the mean results and each one of the iterations results. Some results are also shown in
terminal while the code is executing.
```bash
python3 kNNAlgorithm.py
```

# Run instance reduction on a given data set
In this main file first the reduction technique to be applied and then the data set to be reduced are entered by the
user through the command window. The code will execute the reduction technique defined for the data set given, and will
store the results obtained in a text file (.txt). Some results are also shown in terminal.
```bash
python3 reductionKNNAlgorithm.py
```
