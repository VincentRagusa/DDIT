# DDIT
The Data-Driven Information Theory (DDIT, pronounced "did-it") framework is a set of useful tools for applying information theory to your data.

## Requirements
Right now, the only version of DDIT is a python class. This makes for both ease of use and ease of development. DDIT is developed with python 3.7.3 64-bit and numpy v1.17.2 .

## TODOs
https://pola.rs/

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
There are two main ways to register random variables. The first way is to manually register them, and the second is to automatically register them.
To manually register a column, you can either call DDIT.register_column(),
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
The first allows you to specify a column index of raw_data to load while the second offers you the ability to filter, concatenate, or otherwise manipulate one or several columns before registering the final result.

If your CSV file is organized in such a way that each column *is* a random variable *and* column headers are provided, you can automatically register each column of raw_data as a random variable. The name given to each variable is the corresponding column header.
 ```python
# load data and register each column automatically 
ddit.load_csv("data.csv", header=True, auto_register=True)
```
At any time you can un-register a column. This does not remove the data from DDIT.raw_data.
 ```python
# remove the column whose name is "Input_5" from the list of registered column names 
ddit.remove_column("Input_5")
```
### Calculating Entropies, Information, and Joint Variables
To calculate the entropy of any registered variable:
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
When registering a joint variable, you can optionally specify a name for the new variable with:
 ```python
# register the joint variable "Custom_joint"
ddit.AND("Input_1", "Input_2", new_key="Custom_joint")
```
### Complex Entropy Formulas
Sometimes the labor required to manually register joint variables and calculate shared entropy etc. can be too much. In this case, a function exists to calculate any arbitrary entropy formula that you can give. The acceptable input format is any formula in "standard form" which is here defined as a formula which is in the form "X:Y|Z". There are several mathematical notes to make here:
First, "X:Y|Z", "X:Y", "Y|Z", and "Z" are each examples of formulas in standard form. So is "A:B:C:D|EFGH". In general, standard form is when you express your entropy as a shared entropy (of arbitrarily many variables >= 0) conditional on a joint entropy (of arbitrarily many variables >= 0) and no variable appears more than once.
The formula "" or "|" is undefined (though in a data driven system would always evaluate to 0 anyway if it was).
To get DDIT to evaluate your entropy formula simply call:
 ```python
# calculate and store an entropy given by a formula in standard form
ent = ddit.solve_and_return("X:Y|Z")
```
Alternatively, you can have DDIT solve the formula but not return the result:
 ```python
# calculate an entropy given by a formula in standard form
ddit.recursively_solve_formula("X:Y|Z")
```
To access the resulting entropy value simply:
 ```python
# the result is automatically stored in DDIT.entropies
ent = ddit.entropies["X:Y|Z"]
```

### Generating a system's Venn diagram
To generate the venn diagram of all registered columns:
 ```python
# get the venn diagram of the entire system (all registered columns)
ddit.solve_venn_diagram()
```
To generate the venn diagram for a specific subset of registered columns:
 ```python
# get the venn diagram of X, Y and Z only
ddit.solve_venn_diagram(column_keys=["X","Y","Z"])
```

### Statistical significance of Joint/Shared/Conditional Entropy
With DDIT, you can compute a p-value on any joint, shared, or conditional entropy. These entropies are computed by joining two or more data columns row-wise. For example, the column [0,0,1,1] joined with [0,1,0,1] gives the joint column [(0,0),(0,1),(1,0),(1,1)]. If, for example, we wish to compute the shared entropy between these first two columns, we would find it to be zero. This shared entropy is the result of the particular ordering of the data in these columns. If, instead, the second column was [1,1,0,0] we would have computed the joint column [(0,1),(0,1),(1,0),(1,0)], and the shared entropy would have been 1 bit. Because re-arranging the order of the data can change the entropy computed, it is natural to ask "Is the (joint/shared/conditional) entropy of my data due to a true relationship between the data, or the result of a coincidental ordering of events"? To address this question, and give a quantitative confidence measure of any joint, shared, or conditional entropy, DDIT can compute a permutation p-value.

Usage:
```python
h, P = ddit.solve_with_permutation_pvalue("X:Y|Z")
```
This has the same behavior as solve_and_return but also computes a permutation p-value. The p-value comes from randomly shuffling the order of the data and re-computing the entropy many times to get a distribution of entropy values the data could produce. The proportion of entropies larger than the real entropy is given as a right p-value, and the proportion of entropies smaller than the real entropy is given as a left p-value. The number of times the entropy is re-computed on shuffled data can be set by the 'reps' parameter. More replicates will give better statistics, and the possibility of smaller p-values. The random shuffling can be reproduced by setting the 'seed' parameter, which seeds the python random number generator before permuting the data. 

### Verbose mode
You can set DDIT to verbose mode when you create the object instance or at any time after to get additional printout from many of the DDIT functions. It also enables time stamped messages. This is primarily a debugging tool for developers and end users and does not affect data processing in any way.
Usage:
 ```python
# Verbose mode on object creation
ddit = DDIT(verbose=True)
```
or
 ```python
# Verbose mode after object creation
ddit = DDIT()
ddit.verbose = True
```

### Save Memory mode
You can set DDIT to save memory mode when you create the object instance or at any time after to conserve system memory. There are three modes of operation: `"off"`, `"on"`, and `"lazy"` (`"lazy"` is the default mode). If save memory is off, DDIT will memoize its solved subproblems without any regard for system memory consumption (this is not advisable for large problem sizes). If save memory is turned on, memoization is disabled. If save memory is set to lazy mode, DDIT will memoize solved subproblems until system virtual memory usage is at or above 90% and then the memo is reset.
Usage:
 ```python
# setting save memory to "on" (default setting is "lazy")
ddit = DDIT(save_memory="on")
```
or
 ```python
# changing save memory after object creation (default setting is "lazy")
ddit = DDIT()
ddit.save_memory = "off"
```

### Alternate Entropy measures
By default DDIT calculates all entropies using the maximum-likelihood method, whereby all events' probabilities are estimated by their frequency relative to the total number of observed events. DDIT partially supports the james-stein entropy estimator. It is currently not compatible with DDIT.recursively_solve_formula() and, by extension, also not compatible with DDIT.solve_venn_diagram(). This will be fixed one day.
