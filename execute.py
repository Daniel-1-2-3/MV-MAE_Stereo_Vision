"""
Putting this file in the sif container means that the sif file is built in a way such that
it doesn't need to be rebuilt every time new files are added to the directory and 
changes are made to the training script in main.py.
"""
from main import main
main()