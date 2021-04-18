import mysklearn.myutils as myutils

import copy
import csv 
#from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names) 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            tuple of int: rows, cols in the table

        Notes:
            Raise ValueError on invalid col_identifier
        """
        index = -1
        try:
            index = self.column_names.index(col_identifier)
        except:
            raise ValueError
            return []
        values = []
        for row in self.data:
            val = row[index]
            if include_missing_values or (val != "NA" and val != ""):
                values.append(row[index])
        return values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for i, row in enumerate(self.data):
            for j, x in enumerate(row):
                try:
                    float(x)
                    self.data[i][j] = float(x)
                except:
                    pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for i, row in enumerate(self.data):
            if row in rows_to_drop:
                del self.data[i]

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            self.data = []
            i = 0
            for row in reader:
                if i == 0:
                    self.column_names = row
                else:
                    self.data.append(row)
                i += 1
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.column_names)
            for row in self.data:
                writer.writerow(row)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        duplicates = []
        rows_processed = []
        indices = []
        for key in key_column_names:
            index = self.column_names.index(key)
            indices.append(index)
        for row in self.data:
            matches = True
            try:
                if rows_processed == []:
                    matches = False
                    raise Exception
                for r in rows_processed:
                    matches = True
                    for i in indices:
                        if row[i] != r[i]:
                            matches = False
                    if matches:
                        raise Exception
            except:
                pass
            if matches == True:
                duplicates.append(row)
            else:
                rows_processed.append(row) 
        return duplicates 

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        i = 0
        while i < len(self.data):
            if "NA" in self.data[i]:
                del self.data[i]
            else:
                i += 1

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        
        if col_name == []:
            return 
        col_strings = self.get_column(col_name)
        col_floats = []
        for c in col_strings:
            try:
                f = float(c)
                col_floats.append(f)
            except:
                pass
        if col_floats == []:
            return 
        avg = sum(col_floats) / len(col_floats)
        index = self.column_names.index(col_name)
        for i, row in enumerate(self.data):
            if row[index] == "NA":
                self.data[i][index] = avg


    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        summary_stats = []
        self.convert_to_numeric()
        for row in col_names:
            stats = []
            index = self.column_names.index(row)
            column = self.get_column(row, False)
            if column != []:
                column.sort()
                col_sum = 0
                col_length = 0
                for x in column:
                    try:
                        col_sum += float(x)
                        col_length += 1
                    except:
                        pass
                average = col_sum / col_length
                median = column[int(len(column) / 2)]
                if len(column) % 2 == 0:
                    median = (column[int(len(column)/2)] + column[int(len(column)/2) - 1]) / 2
                stats.append(row)
                stats.append(min(column))
                stats.append(max(column))
                stats.append((max(column)+min(column))/2)
                stats.append(average)
                stats.append(median)
                summary_stats.append(stats)
        new_table = MyPyTable(col_names, summary_stats)
        return new_table



        return MyPyTable() # TODO: fix this

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_data = []
        key_indices_1 = []
        key_indices_2 = []

        # compile the new table
        for key in key_column_names:
            key_indices_1.append(self.column_names.index(key))
            key_indices_2.append(other_table.column_names.index(key))
        for i,row in enumerate(self.data):
            for j,row2 in enumerate(other_table.data):
                # check to see if we can do an inner join
                can_join = True
                try:
                    for k, key1 in enumerate(key_indices_1):
                        if row[key1] != row2[key_indices_2[k]]:
                            can_join = False
                            raise Exception
                except:
                    pass
                # if we can join, then join
                if can_join:
                    new_row = copy.deepcopy(row)
                    for l,x in enumerate(row2):
                        if l not in key_indices_2:
                            new_row.append(x)
                    new_data.append(new_row)
        
        # compile the new header
        new_header = self.column_names
        for x in other_table.column_names:
            if x not in key_column_names:
                new_header.append(x)
                
        return MyPyTable(new_header, new_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_data = []
        key_indices_1 = []
        key_indices_2 = []

        
        for key in key_column_names:
            key_indices_1.append(self.column_names.index(key))
            key_indices_2.append(other_table.column_names.index(key))

        # do a left outer join
        for i,row in enumerate(self.data):
            did_join = False
            for j,row2 in enumerate(other_table.data):
                # check to see if we can do an inner join
                can_join = True
                try:
                    for k, key1 in enumerate(key_indices_1):
                        if row[key1] != row2[key_indices_2[k]]:
                            can_join = False
                            raise Exception
                except:
                    pass
                # if we can join, then join
                if can_join:
                    new_row = copy.deepcopy(row)
                    for l,x in enumerate(row2):
                        if l not in key_indices_2:
                            new_row.append(x)
                    new_data.append(new_row)
                    did_join = True
            if not did_join:
                new_row = copy.deepcopy(row)
                for l,x in enumerate(row2):
                    if l not in key_indices_2:
                        new_row.append("NA")
                new_data.append(new_row)
            
        # do a right outer join minus the corresponding data from the left
        for j,row2 in enumerate(other_table.data):
            did_join = False
            for i,row in enumerate(self.data):
                # check to see if we can do an inner join
                can_join = True
                try:
                    for k, key1 in enumerate(key_indices_1):
                        if row[key1] != row2[key_indices_2[k]]:
                            can_join = False
                            raise Exception
                except:
                    pass
                if can_join:
                    # since we are not doing a complete right outer join
                    # dont add the corresping data 
                    did_join = True
            if not did_join:
                new_row = []
                for l,x in enumerate(row):
                    if l not in key_indices_1:
                        new_row.append("NA")
                    else:
                        new_row.append(row2[key_indices_2[key_indices_1.index(l)]])
                for l,x in enumerate(row2):
                    if l not in key_indices_2:
                        new_row.append(x)
                new_data.append(new_row)

        
        # compile the new header
        new_header = self.column_names
        for x in other_table.column_names:
            if x not in key_column_names:
                new_header.append(x)
                
        return MyPyTable(new_header, new_data)
        return MyPyTable() # TODO: fix this