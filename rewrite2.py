"""
A data-driven information theory framework.
Allows the computation and composition of Shannon entropy and information.
Data is stored and manipulated as lists of co-occurring variable states.
Entropy expressions are evaluated recursively until they can be resolved.

Vincent Ragusa 2019-2025
"""
from collections import Counter
from csv import reader

from numpy import array, dot, log2


def make_expression(shared_parts:list[str],condition:list[str]) -> str:
    """Makes a valid entropy expression from the given components.

    Args:
        shared_parts (list[str]): The variables on the shared side of the condition (one or more)
        condition (list[str]): The variables on the joint side of the condition (zero or more)

    Returns:
        str: The corresponding entropy expression
    """
    return ":".join(shared_parts)+ ("|" if condition else "") + "&".join(condition)


def transpose(list_of_list:list[list]) -> list[list]:
    """Transposes a list of list (e.g., rows -> cols)

    Args:
        list_of_list (list[list]): Original list of list

    Returns:
        list[list]: Transposed list of list.
    """
    return list(zip(*list_of_list))


class DDIT():
    """Data Driven Information Theory
    """
    def __init__(self):
        self.data:dict[str,list] = {}
        self.num_events = None
        self._temp_data:dict[str,list] = {}
        self._use_temp_data:bool = False


    def read_csv(self, path_to_csv:str, assume_column_headers:bool=True) -> None:
        """reads data in from a csv file.

        Args:
            path_to_csv (str): Path to the csv file on disk.
            assume_column_headers (bool, optional): 
                Read the first row of the csv as variable names. If False,
                each column is identified by a unique integer. Defaults to True.
        """
        header = None
        data = None

        with open(path_to_csv, 'r', encoding="utf-8") as f:
            r = reader(f)
            if assume_column_headers:
                header = next(r)
            data = [row for row in r] #TODO: = list(r)

        data_t = transpose(data)
        self.num_events = len(data_t[0])

        for i, variable in enumerate(data_t):
            if assume_column_headers:
                self.data[header[i]] = variable
            else:
                self.data[i] = variable


    def variable_names(self) -> list[str]:
        """Returns all known variable names.

        Returns:
            list[str]: List of variable names.
        """
        return list(self.data.keys())


    def entropy(self, variable_name:str, data:list = None) -> float:
        """Computes the Shannon Entropy of the named variable.

        Args:
            variable_name (str): Name of stored variable.
            data (list, optional): List of symbols.

        Returns:
            float: Shannon Entropy of variable. Defaults to None.
        """
        if data is None:
            event_counts = Counter(self.data[variable_name]).values()
        else:
            event_counts = Counter(data).values()
        event_probabilities = array([count/self.num_events for count in event_counts])
        return 0.0 - dot(event_probabilities, log2(event_probabilities))


    def evaluate_expression(self, entropy_expression:str) -> float:
        """Computes the value of arbitrary entropy expressions in 'normal form'.
        Expressions are decomposed using the following identities:\n
        A:B:C|D = A:B|D - A:B|C&D,\n
        A:B|C = A|C - A|B&C,\n
        A|B = A&B - B.\n
        Together, these expressions allow for a recursive decomposition of any
        formula into the sum of joint entropies.

        Args:
            entropy_expression (str): An entropy expression in normal form.
                Normal form is 'X: ... :Y|Z', where X, Y, and Z can be single or joint
                random variables and there can be arbitrarily many 'shared' variables.
                No single variable can appear more than once. Use joint variable aliases
                to prevent multiple appearances of a variable if necessary. It is never
                necessary.

        Returns:
            float: The entropy of the given expression.
        """
        split_conditional = entropy_expression.split("|")
        shared_parts = sorted(split_conditional[0].split(":"))
        condition = sorted([var for chunk in split_conditional[1:] for var in chunk.split("&")])

        if len(shared_parts) > 1:
            #A:B:C|D = A:B|D - A:B|C&D
            #A:B|C = A|C - A|B&C
            shifted_var = shared_parts.pop().split("&")
            lhs = make_expression(shared_parts,condition)
            condition.extend(shifted_var)
            condition.sort()
            rhs = make_expression(shared_parts,condition)
            return self.evaluate_expression(lhs) - self.evaluate_expression(rhs)

        if condition:
            #A|B = A&B - B
            lhs = make_expression(["&".join(sorted(condition+shared_parts[0].split("&")))],[])
            rhs = make_expression(["&".join(condition)],[])
            return self.evaluate_expression(lhs) - self.evaluate_expression(rhs)

        variables = shared_parts[0].split("&")

        if len(variables) > 1:
            # A&B
            sorted_names = sorted(variables)
            return self.entropy(None, data=tuple(zip(*[self.data[name] for name in sorted_names])))

        # A
        return self.entropy(variables[0])


if __name__ == "__main__":

    ddit = DDIT()

    ddit.read_csv("./xor_data.csv")

    print(ddit.data)

    print(ddit.evaluate_expression("X:Y|Z"))
