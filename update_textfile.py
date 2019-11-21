"""
Module to automatically create the 100 exercises (with or without hints and answers)
as a textfile format (markdown format).
"""

from .questsions_dict import qha


HEADER = "DO NOT MODIFY. \nFile automatically created. To modify change the content of questions_dict.py and then" \
         "re create via the python script 'create_as_textfile.py'. Search the documentation for more info."


def to_markdown(destination_filename, with_ints=False, with_answers=False):
    pass


def to_rst(destination_filename, with_ints=False, with_answers=False):
    pass


def cli():
    pass
