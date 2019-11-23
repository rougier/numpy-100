import os
import nbformat as nbf
import mdutils

import data_source as ds


def create_jupyter_notebook(destination_filename='numpy_100.ipynb'):
    """ Programmatically create jupyter notebook with the questions (and hints and solutions if required)
    saved under data_source.py """

    # Create cells sequence
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    # - Add header:
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.HEADER))
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.SUB_HEADER))
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.JUPYTER_INSTRUCTIONS))

    # - Add initialisation
    nb['cells'].append(nbf.v4.new_code_cell('%run initialise.py'))

    # - Add questions and empty spaces for answers
    for n in range(1, 101):
        nb['cells'].append(nbf.v4.new_markdown_cell(f'#### {n}. ' + ds.QHA[f'q{n}']))
        nb['cells'].append(nbf.v4.new_code_cell(""))

    # Delete file if one with the same name is found
    if os.path.exists(destination_filename):
        os.remove(destination_filename)

    # Write sequence to file
    nbf.write(nb, destination_filename)


def create_jupyter_notebook_random_question(destination_filename='numpy_100_random.ipynb'):
    """ Programmatically create jupyter notebook with the questions (and hints and solutions if required)
    saved under data_source.py """

    # Create cells sequence
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    # - Add header:
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.HEADER))
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.SUB_HEADER))
    nb['cells'].append(nbf.v4.new_markdown_cell(ds.JUPYTER_INSTRUCTIONS_RAND))

    # - Add initialisation
    nb['cells'].append(nbf.v4.new_code_cell('%run initialise.py'))
    nb['cells'].append(nbf.v4.new_code_cell("pick()"))

    # Delete file if one with the same name is found
    if os.path.exists(destination_filename):
        os.remove(destination_filename)

    # Write sequence to file
    nbf.write(nb, destination_filename)


def create_markdown(destination_filename='numpy_100', with_hints=False, with_solutions=False):
    # Create file name
    if with_hints:
        destination_filename += '_with_hints'
    if with_solutions:
        destination_filename += '_with_solutions'

    # Initialise file
    mdfile = mdutils.MdUtils(file_name=destination_filename)

    # Add headers
    mdfile.write(ds.HEADER)
    mdfile.write(ds.SUB_HEADER)

    # Add questions (and hint or answers if required)
    for n in range(1, 101):
        mdfile.new_header(title=f"{n}. {ds.QHA[f'q{n}']}", level=4)
        if with_hints:
            mdfile.write(f"`{ds.QHA[f'h{n}']}`")
        if with_solutions:
            mdfile.insert_code(ds.QHA[f'a{n}'], language='python')

    # Delete file if one with the same name is found
    if os.path.exists(destination_filename):
        os.remove(destination_filename)

    # Write sequence to file
    mdfile.create_md_file()


def create_rst(destination_filename, with_ints=False, with_answers=False):
    # TODO: use rstdoc python library.
    #  also see possible integrations with https://github.com/rougier/numpy-100/pull/38
    pass


if __name__ == '__main__':
    create_jupyter_notebook()
    create_jupyter_notebook_random_question()
    create_markdown()
    create_markdown(with_hints=False, with_solutions=True)
    create_markdown(with_hints=True, with_solutions=False)
    create_markdown(with_hints=True, with_solutions=True)
