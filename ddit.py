"""
A data-driven information theory framework.
Allows the computation and composition of shannon entropy and mutual information.
Data is stored and manipulated as columns/rows of variable states
entropy is calculated from collapsing these tables

Vincent Ragusa 2019-2025
"""
from collections import Counter
from csv import reader
from datetime import datetime
from random import seed, shuffle

from numpy import array, dot, isclose, log2


class DDIT:
    """A data driven information theory framework. 
    """

    def __init__(self, verbose:bool=False):
        self.raw_data = None
        self.labels = None
        self._columns:dict[str:tuple[str]] = {}
        self.column_keys:set[str] = set()
        self.verbose:bool = verbose
        self.events_recorded:int = None


    def _get_column(self, key:str)->tuple[str]:
        return self._columns[key]


    def _set_column_tuple(self, key:str, value_list:tuple[str])->None:
        self._columns[key] = value_list
        self.column_keys.add(key)


    def remove_column(self,key:str)->None:
        """Deletes a column from the registered columns.

        Args:
            key (str): Name of column to delete.
        """
        if self.verbose:
            print(f"{str(datetime.now())} Deleting {key} from memory ...")
        del self._columns[key]
        self.column_keys.remove(key)


    def _columns_contains(self, key:str)->bool:
        return key in self.column_keys


    def _columns_empty(self)->bool:
        return False if self.column_keys else True


    def load_csv(self, file_path:str, header:bool=False, auto_register:bool=False)->None:
        """Loads data from a csv.
        Can automatically detect column headers and register those columns.

        Args:
            file_path (str): path to data file
            header (bool, optional): does the data file have a header? Defaults to False.
            auto_register (bool, optional): should DDIT register column headers? Defaults to False.
        """
        if self.verbose:
            print(f"{str(datetime.now())} Loading data from {file_path} ...")
        with open(file_path, 'r', encoding="utf-8") as csv_file:
            rdr = reader(csv_file)
            if header:
                self.labels = next(rdr)
                self.raw_data = [tuple(row) for row in rdr]
                if auto_register:
                    for col, label in enumerate(self.labels):
                        self.register_column(label,col)
            else:
                self.raw_data = [row for row in rdr]
                if auto_register:
                    print("WARNING you must manually register columns if no header is provided. "
                    "Skipping auto-register...")


    def register_column(self, key:str, col_index:int)->None:
        """adds the column name to DDIT's column registry
        and links the corresponding data.

        Args:
            key (str): column name
            colIndex (int): index in rawData to locate data (only when load_csv is used)
        """
        if self.verbose:
            print(f"{str(datetime.now())} Registering {col_index} as {key} ...")

        new_column = tuple(row[col_index] for row in self.raw_data)

        len_col = len(new_column)
        if self.events_recorded is None:
            self.events_recorded = len_col
        assert len_col == self.events_recorded # all columns must be the same length (rows)

        if self._columns_contains(key):
            print(f"WARNING: column \"{key}\" has already been registered. Overwriting...")
        self._set_column_tuple(key,new_column)


    def register_column_tuple(self, key:str, col:tuple[str])->None:
        """Registers a column using the provided data.

        Args:
            key (str): Name of column.
            col (tuple[str]): Data of column. Must be the same length as existing columns.
        """
        if self.verbose:
            print(f"{str(datetime.now())} Registering custom column as {key} ...")

        len_col = len(col)
        if self.events_recorded is None:
            self.events_recorded = len_col
        assert len_col == self.events_recorded # all columns must be the same length (rows)

        if self._columns_contains(key):
            print(f"WARNING: column \"{key}\" has already been registered. Overwriting...")

        self._set_column_tuple(key,col)


    def print_columns(self)->None:
        """Prints the registered columns.
        """
        if self._columns_empty():
            print("No registered columns to print.")
        else:
            for key in self.column_keys:
                print(key, self._get_column(key))
        print()


    def join_and_register(self, col1:str, col2:str, new_key:str=None)->None:
        """Joins two columns and adds their joint data to the registered columns.

        Args:
            col1 (str): name of first column
            col2 (str): name of second column
            new_key (str, optional): Name of new column. Defaults to None.
        """
        if self.verbose:
            print(f"{str(datetime.now())} Creating joint distribution: {col1}&{col2} ...")
        if not self._columns_contains(col1):
            print(f"ERROR column \"{col1}\" is not registered!")
            return
        if not self._columns_contains(col2):
            print(f"ERROR column \"{col2}\" is not registered!")
            return
        if new_key is not None:
            key = new_key
        else:
            key = col1 + "&" + col2
        new_col = list(zip(self._get_column(col1),self._get_column(col2)))
        self.register_column_tuple(key,new_col)


    def entropy(self, col_name:str, column_data:list[str] = None)->float:
        """Computes the Shannon Entropy of the specified data.

        Args:
            col_name (str): Name of registered column to retrieve data from. 
            column_data (list[str], optional): Data to compute entropy with.
                Defaults to None.

        Returns:
            float: The entropy of the given data.
        """
        if not self._columns_contains(col_name) and column_data is None:
            print(f"ERROR column \"{col_name}\" is not registered!")
            exit(1)

        if column_data is not None:
            events_data = column_data
        else:
            events_data = self._get_column(col_name)

        event_counts = Counter(events_data).values()
        p_vector = array([count/ self.events_recorded for count in event_counts])
        return 0.0 - dot(p_vector, log2(p_vector))


    def information(self, col1:str, col2:str)->float:
        """Computes the information (shared entropy) between two variables.

        Args:
            col1 (str): Name of first variable.
            col2 (str): Name of second variable.

        Returns:
            float: Information (shared entropy) between the two variables.
        """
        and_key = col1 + "&" + col2
        if not self._columns_contains(and_key):
            if self.verbose:
                print(f"{str(datetime.now())} registering column \"{and_key}\"...")
            self.join_and_register(col1,col2)
        return self.entropy(col1) + self.entropy(col2) - self.entropy(and_key)


    def _venn_gen_power_set(self, variables, current=None):
        if current is None:
            current = []
        if variables:
            a = self._venn_gen_power_set(variables[1:],current=current)
            b = self._venn_gen_power_set(variables[1:], current=current + [variables[0]])
            return a + b
        return [current]


    def _venn_make_formula(self, subset, column_keys):
        if subset:
            subset.sort()
            remainder = sorted(list(set(column_keys) - set(subset)))

            s=""
            s = ":".join(subset)
            if remainder:
                s += "|"
            s += "&".join(remainder)
            return s
        return ""

    # ~O(S) where S is number of shared variables on LHS of |
    def recursively_solve_formula(self, formula:str)->float:
        """Computes the value of arbitrary entropy expressions in 'normal form'.
        Expressions are decomposed using the following identities:\n
        A:B:C|D = A:B|D - A:B|C&D,\n
        A:B|C = A|C - A|B&C,\n
        A|B = A&B - B.\n
        Together, these expressions allow for a recursive decomposition of any
        formula into the sum of joint entropies.

        Args:
            formula (str): An entropy expression in normal form.
                Normal form is 'X: ... :Y|Z', where X, Y, and Z can be single or joint
                random variables and there can be arbitrarily many 'shared' variables.
                No single variable can appear more than once. Use joint variable aliases
                to prevent multiple appearances of a variable if necessary. It is never
                necessary.

        Returns:
            float: The entropy of the given expression.
        """
        if "|" in formula:
            # formula is a conditional
            halves = formula.split("|")
            if ":" in halves[0]:
                #formula is a conditional with shared entropy on the lhs
                shared = halves[0].split(":")
                left_formula = ":".join(shared[1:]) + "|" + halves[1]
                right_formula = ":".join(shared[1:]) + "|" + "&".join(sorted(
                    halves[1].split("&") + [shared[0]]))#sorted to keep keys unique
                #A:B|C = B|C - B|AC
            else:
                #formula is a conditional of only joints
                left_formula = "&".join(sorted(halves[1].split("&") + halves[0].split("&")))
                right_formula = halves[1]
                #A|B = AB-B
            return self.recursively_solve_formula(
                left_formula) - self.recursively_solve_formula(right_formula)

        if ":" in formula:
            #formula is shared only; treat as special case of above case
            shared = formula.split(":")
            left_formula = ":".join(shared[1:])
            right_formula = ":".join(shared[1:]) + "|" + shared[0]
            #A:B = B - B|A
            return self.recursively_solve_formula(
                left_formula) - self.recursively_solve_formula(right_formula)

        # formula is only a joint; calculate from data
        variables = formula.split("&")
        if len(variables) == 1:
            return self.entropy(formula)

        joint_data = tuple(zip(*[self._get_column(v) for v in variables]))
        return self.entropy(formula, column_data=joint_data)


    #TODO: return values rather than just print
    def solve_venn_diagram(self, column_keys:list[str]=None)->None:
        """Computes all entropy expressions comprising the entropy Venn diagram
        of the variables provided.

        Args:
            column_keys (list[str], optional): List of variable names to include
                in the Venn diagram. If None is passed, all registered columns
                are included in the diagram. Defaults to None.
        """
        if column_keys is None:
            column_keys = list(self.column_keys)
        if self.verbose:
            print(f"{str(datetime.now())} Generating power set of {column_keys}...")
        power_set = self._venn_gen_power_set(column_keys)
        if self.verbose:
            print(f"{str(datetime.now())} Generating venn diagram...")
        else:
            print("Generating venn diagram...")
        for i, subset in enumerate(power_set):
            if i > 0:
                formula = self._venn_make_formula(subset, column_keys)
                ent = self.recursively_solve_formula(formula)
                if self.verbose:
                    print(f"{str(datetime.now())} {i} {formula} {ent}")
                else:
                    print(f"{i} {formula} {ent}")


    def greedy_condition_adder(self, focal_var:str,
                               other_var_list:list[str],
                               max_conditions:int=None) -> list[str]:
        """A greedy algorithm for feature selection. Iteratively conditions on the variable that 
        reduces the most entropy in the target variable. This algorithm is NOT guaranteed to
        find the smallest explanatory set.

        Args:
            focal_var (str): The variable we wish to fully explain.
                other_var_list (list[str]): The set of variables used to explain the focal variable.
                max_conditions (int, optional): A maximum number of conditional variables to return.
                Defaults to None.

        Returns:
            list[str]: list of variables identified as explanatory over the focal var.
        """
        #most entropy explainable is given by joint everything
        print(f"Finding Minimal explanatory set for {focal_var} (GCA)...")
        formula = f"{focal_var}|{'&'.join(other_var_list)}"
        target = self.recursively_solve_formula(formula)
        chosen = []
        if max_conditions is None:
            max_conditions = len(other_var_list)
        time = 1
        while len(chosen) < max_conditions:
            if chosen:
                all_conditioned_entropies = [self.recursively_solve_formula(
                    f"{focal_var}|{'&'.join(chosen+[other])}") for other in other_var_list]
                best_entropy = min(all_conditioned_entropies)
                best_other = other_var_list[all_conditioned_entropies.index(best_entropy)]
                print(f"{time} |{best_other}", best_entropy)
            else:
                all_conditioned_entropies = [self.recursively_solve_formula(
                    f"{focal_var}|{other}") for other in other_var_list]
                best_entropy = min(all_conditioned_entropies)
                best_other = other_var_list[all_conditioned_entropies.index(best_entropy)]
                print(f"{time} |{best_other}", best_entropy)
            chosen.append(best_other)
            other_var_list.remove(best_other)
            time += 1
            if isclose(best_entropy,target):
                return chosen
        print("WARNING: greedy_condition_adder did not meet the target!")
        return chosen


    def smallest_explanatory_set(self, focal_var:str, other_vars:list[str],
                                 _keep_vars:list[str]=None, _best:list[str]=None,
                                 _target:float=None) -> list[str]:
        """A branch-and-bound algorithm for finding the smallest set of variables needed to
        explain the focal variable.

        Args:
            focal_var (str): Name of the variable to explain.
            other_vars (list[str]): Names of variables to include as explainers. 
            _keep_vars (list[str], optional): Variables used in the current hypothesis.
                Defaults to None.
            _best (list[str], optional): Best set of explainers so far. Defaults to None.
            _target (float, optional): target entropy. Defaults to None.

        Returns:
            list[str]: List of explainer variables.
        """
        #first call init
        if _keep_vars is None:
            _keep_vars = []
            # sort other vars by order 1 information, as a heuristic
            other_vars.sort(key = lambda x: self.recursively_solve_formula(
                f"{focal_var}:{x}"),reverse=True)
            #a small fully explanatory set is found by greedy algorithm
            _best = self.greedy_condition_adder(focal_var,other_vars)
            #most entropy explainable is given by joint everything
            formula = f"{focal_var}|{'&'.join(other_vars)}"
            _target = self.recursively_solve_formula(formula)
            print("TARGET",_target)

        #can we do better than the reported best, with the choices available?
        #if adding another variable would make us the same size as the best,
        #there's no point looking further.
        if len(_keep_vars)+1 == len(_best):
            return _best

        #sort other vars to speed up tree search
        other_vars.sort(key= lambda other: self.recursively_solve_formula(
            f"{focal_var}|{'&'.join(_keep_vars+[other])}") )

        #if len(otherVars) == 0, skip this loop and return best.
        for i in range(len(other_vars)):
            # each loop evaluates one of the choices. The recursive call below always
            # includes the choice. thus, we check including and excluding each variable.
            # We also eliminate repeated sets by keeping the choices ordered.
            # i.e., if X|AB is tested in one branch, X|BA will not be tested in another branch.
            choices = other_vars[i:]
            #we first check if including every available choice is worse than the target.
            #if including everything is worse, we cannot do better than the reported best.
            formula = f"{focal_var}|{'&'.join(_keep_vars+choices)}"
            bound_test = self.recursively_solve_formula(formula)
            if not isclose(bound_test,_target):
                return _best
            #we now test if including only our first choice reaches the target.
            formula = f"{focal_var}|{'&'.join(_keep_vars+[choices[0]])}"
            ent = self.recursively_solve_formula(formula)
            # print(f,entropy) #DEBUG PRINT
            # if including our choice meets the target, (and we are smaller than the best,)
            # return the new best. note, including more variables cannot improve things further,
            # and we cannot break ties in a meaningful way. therefore, making a recursive call
            # is pointless, and considering more iterations of this loop is wasteful.
            if isclose(ent,_target):# and len(keepVars)+1 < len(best):
                print("Rejoice, A new best!",_keep_vars+[choices[0]],"\n")
                return _keep_vars+[choices[0]]
            #if we have not met the target, we must include another variable. Unsure of which
            # to include, we ask SES. the variable we are considering is added to the keepVars
            # so that SES will consider it as given. the best set and target are passed to keep
            # SES as up-to-date as possible
            if len(_keep_vars) + 2 < len(_best):
                _best = self.smallest_explanatory_set(
                    focal_var, choices[1:], _keep_vars+[choices[0]], _best, _target)
        #after evaluating all choices, return the best.
        return _best


    def _get_var_list(self,formula:str)->list[str]:
        if "|" in formula:
            parts = formula.split("|")
            return self._get_var_list(parts[0]) + self._get_var_list(parts[1])
        if ":" in formula:
            return formula.split(":")
        if "&" in formula:
            return formula.split("&")
        return [formula]


    def _tokenize_formula(self,formula:str)->list[str]:
        if "|" in formula:
            parts = formula.split("|")
            return self._tokenize_formula(parts[0]) + ["|"] + self._tokenize_formula(parts[1])
        if ":" in formula:
            variables = formula.split(":")
            return_list = []
            for var in variables:
                return_list.append(var)
                return_list.append(":")
            return return_list[:-1] #remove last :
        if "&" in formula:
            variables = formula.split("&")
            return_list = []
            for var in variables:
                return_list.append(var)
                return_list.append("&")
            return return_list[:-1] #remove last &
        return [formula]


    def _get_temp_formula(self, formula:str, rep_num:int)->str:
        tokens = self._tokenize_formula(formula)
        new_formula = ""
        for token in tokens:
            if token not in [":","|","&"]:
                new_formula += f"PR{rep_num}_{token}"
            else:
                new_formula += token
        return new_formula

    def solve_with_permutation_pvalue(self,
                                      formula:str,
                                      reps:int=100,
                                      rseed:int=None) -> tuple[float,tuple[float,float]]:
        """Evaluates an entropy expression and returns a p-value for the result.

        Args:
            formula (str): An entropy expression in normal form.
            reps (int, optional): Number of permutation tests to run. Higher
                values result in more accurate p-values, but run longer.
                Defaults to 100.
            rseed (int, optional): The random seed. Can be set to reproduce results.
                Defaults to None.

        Returns:
            tuple[float,tuple[float,float]]: _description_
        """
        if rseed is not None:
            seed(rseed)
        variables = self._get_var_list(formula)
        ent = self.recursively_solve_formula(formula)
        #collect bootstrap distribution
        distribution = []
        for rep in range(reps):
            #create clone columns
            # newFormula = formula
            new_formula = self._get_temp_formula(formula,rep)
            for var in variables:
                shuffled_values = list(self._columns[var])
                shuffle(shuffled_values)
                self.register_column_tuple(f"PR{rep}_{var}",tuple(shuffled_values))
                # newFormula = newFormula.replace(var,"BS{}_{}".format(rep,var))
            #get entropy
            distribution.append(self.recursively_solve_formula(new_formula))
            #remove columns
            for key in list(self.column_keys):
                if key.startswith(f"PR{rep}_"):
                    self.remove_column(key)
        #estimate p-value
        p_left = 0
        p_right = 0
        for value in distribution:
            if value >= ent:
                p_right += 1
            if value <= ent:
                p_left += 1
        p_left /= reps
        p_right /= reps

        return ent, (p_left,p_right)



if __name__ == "__main__":

    # create an instance of the class
    ddit = DDIT(verbose=True)

    # auto register columns based on CSV headers
    ddit.load_csv("xor_data.csv", header=True, auto_register=True)

    # display registered columns
    ddit.print_columns()

    # get the venn diagram of the system
    ddit.solve_venn_diagram(column_keys=["X","Y","Z"])

    # calculate an arbitrary entropy given in standard form
    entropy = ddit.recursively_solve_formula("X:Y|Z")
    print("The entropy of X:Y|Z is ", entropy)
