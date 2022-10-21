# Set display options when printing pandas DataFrames
import pandas as pd

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('max_colwidth', 100)  # Display up to this number of characters
pd.set_option('display.width', 10000)  # Display up to this number of characters on the same line
pd.set_option('max_seq_item', 5)  # Display up to this number of items in a sequence
pd.set_option('display.precision', 2)  # Display up to this number of decimal digits
