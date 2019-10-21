# DDIT
The Data-Driven Information Theory (DDIT, pronounced "did-it") framework is a set of useful tools for applying information theory to your data.

## Python
Right now, the only version of DDIT is a python class. This makes for both ease of use and ease of developement. DDIT is developed with python 3.7.3 64-bit and numpy v1.17.2 .

## Usage
The following sections describe how to use DDIT.

### Initializing DDIT
To begin, simply import DDIT.py into your python script. Once imported, create a DDIT object:
```python
# import DDIT class
from DDIT import DDIT

# create an instance of the class
ddit = DDIT()
```

### Loading Data
DDIT assumes you can provide your data in a CSV format. If you can, use the DDIT load_csv() function to load it.
```python
# load data that has no column headers
ddit.load_csv("data.csv")
```
By default, DDIT assumes there are no column headers and will load the first row as data. If your data has column headers instead use:
```python
# load data that has column headers
ddit.load_csv("data.csv", header=True)
```
Data that has been loaded is stored in DDIT.raw_data as a list of lists (RC addressable). If the header flag was set, the column titles are stored in DDIT.labels as a list of strings.
DDIT.raw_data is used as a staging ground for you to construct your random variables (hereafter also known as "columns").

### Registering Columns (Random Variables)
There are two main ways to regester random variables. The first way is to manually register them, and the second is to automatically regester them.
To manually regester a column, you can either call DDIT.register_column(),
```python
# register column 0 of raw_data as "Input_1"
ddit.register_column("Input_1", 0)
```
 or DDIT.register_column_list().
 ```python
# register column 1 of raw_data as "Input_2" via custom column construction
custom_column = [row[1] for row in ddit.raw_data]
ddit.register_column_list("Input_2", custom_column)
```
The first allows you to specify a column index of raw_data to load while the second offers you the ability to filter, concatinate, or otherwise manipulate one or several columns before regestering the final result.

If your CSV file is organized in such a way that each column *is* a random variable *and* column headers are provided, you can automatically regester each column of raw_data as a random variable. The name given to each variable is the corresponding column header.
 ```python
# load data and regester each column automatically 
ddit.load_csv("data.csv", header=True, auto_register=True)
```
### Calculating Entropies, Information, and Joint Variables
To calculate the entropy of any regestered variable:
 ```python
# Calculate the entropy of "Input_1"
e1 = ddit.H("Input_1")
```
To calculate the shared entropy (Information) between two columns:
 ```python
# Calculate The information between "Input_1" and "Input_2"
i12 = ddit.I("Input_1", "Input_2")
```
To register a joint variable:
 ```python
# register the joint variable "Input_1&Input_2"
ddit.AND("Input_1", "Input_2")
```
### Complex Entropy Formulas
Sometimes the labor required to manually regester joint variables and calculate shared entropy etc. can be too much. In this case, a function exists to calculate any arbetrary entropy formula that you can give. The acceptable input format is any formula in "standard form" which is here defined as a formula which is in the form "X:Y|Z&W"
