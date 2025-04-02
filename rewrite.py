"""
DDIT: A D_ata-D_riven I_nformation T_heory framework.

Vincent Ragusa 2019-2025
"""
import polars as pl  # pip install --user polars


class DDIT:
    """DDIT class contains all functionality for working with data
        and computing information theoretic quantities.
    """
    def __init__(self, verbose:bool=False) -> None:
        self.verbose:bool = verbose
        self.df:pl.DataFrame = None


    def read_csv(self,path_to_csv:str) -> None:
        """loads data from the specified csv file.

        Args:
            path_to_csv (str): path to csv file 
        """
        self.df = pl.read_csv(path_to_csv)

        if self.verbose:
            print(f"Reading data from {path_to_csv}")
            print(self.df)


    def entropy(self,column_name:str) -> pl.DataFrame:
        """Computes the Shannon entropy of the column specified.

        Args:
            column_name (str): column identifier 

        Returns:
            pl.DataFrame: a dataframe containing the Shannon Entropy
        """
        if self.verbose:
            print(f"Computing the entropy of {column_name}")
        alias = f"H({column_name})"
        select = pl.col(column_name) .cast(pl.Utf8) .unique_counts() .entropy(base=2) .alias(alias)
        return self.df.select(select)


    def join(self,column_name_1:str, column_name_2:str) -> None:
        """Joins together two columns so their joint entropy can be computed at a later time.
        The joint column is added to the set of existing columns.

        Args:
            column_name_1 (str): First column name
            column_name_2 (str): Second column name
        """
        if self.verbose:
            print(f"Joining columns {column_name_1} and {column_name_2}")
        alias = f"{column_name_1}&{column_name_2}"
        with_columns = pl.struct(column_name_1,column_name_2) .alias(alias)
        self.df = self.df.with_columns(with_columns)


if __name__ == "__main__":
    ddit = DDIT(verbose=True)
    ddit.read_csv("./xor_data.csv")

    e = ddit.entropy("X")
    print(e)

    ddit.join("X","Y")
    print(ddit.df)
    print(ddit.entropy("X&Y"))

    ddit.join("X&Y","Z")
    print(ddit.df)
    print(ddit.entropy("X&Y&Z"))
