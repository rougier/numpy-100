import os
import nbformat as nbf
import mdutils


def ktx_to_dict(input_file, keystarter='<'):
    """ parsing keyed text to a python dictionary. """
    answer = dict()

    with open(input_file, 'r+', encoding='utf-8') as f:
        lines = f.readlines()

    k, val = '', ''
    for line in lines:
        if line.startswith(keystarter):
            k = line.replace(keystarter, '').strip()
            val = ''
        else:
            val += line

        if k:
            answer.update({k: val.strip()})

    return answer


def dict_to_ktx(input_dict, output_file, keystarter='<'):
    """ Store a python dictionary to a keyed text"""
    with open(output_file, 'w+') as f:
        for k, val in input_dict.items():
            f.write(f'{keystarter} {k}\n')
            f.write(f'{val}\n\n')


HEADERS = ktx_to_dict(os.path.join('source', 'headers.ktx'))
QHA = ktx_to_dict(os.path.join('source', 'exercises100.ktx'))


def create_jupyter_notebook(destination_filename='100_Numpy_exercises.ipynb'):
    """ Programmatically create jupyter notebook with the questions (and hints and solutions if required)
    saved under source files """

    # Create cells sequence
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    # - Add header:
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["header"]))
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["sub_header"]))
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["jupyter_instruction"]))

    # - Add initialisation
    nb['cells'].append(nbf.v4.new_code_cell('%run initialise.py'))

    # - Add questions and empty spaces for answers
    for n in range(1, 101):
        nb['cells'].append(nbf.v4.new_markdown_cell(f'#### {n}. ' + QHA[f'q{n}']))
        nb['cells'].append(nbf.v4.new_code_cell(""))

    # Delete file if one with the same name is found
    if os.path.exists(destination_filename):
        os.remove(destination_filename)

    # Write sequence to file
    nbf.write(nb, destination_filename)


def create_jupyter_notebook_random_question(destination_filename='100_Numpy_random.ipynb'):
    """ Programmatically create jupyter notebook with the questions (and hints and solutions if required)
    saved under source files """

    # Create cells sequence
    nb = nbf.v4.new_notebook()

    nb['cells'] = []

    # - Add header:
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["header"]))
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["sub_header"]))
    nb['cells'].append(nbf.v4.new_markdown_cell(HEADERS["jupyter_instruction_rand"]))

    # - Add initialisation
    nb['cells'].append(nbf.v4.new_code_cell('%run initialise.py'))
    nb['cells'].append(nbf.v4.new_code_cell("pick()"))

    # Delete file if one with the same name is found
    if os.path.exists(destination_filename):
        os.remove(destination_filename)

    # Write sequence to file
    nbf.write(nb, destination_filename)


def create_markdown(destination_filename='100_Numpy_exercises', with_hints=False, with_solutions=False):
    # Create file name
    if with_hints:
        destination_filename += '_with_hints'
    if with_solutions:
        destination_filename += '_with_solutions'

    # Initialise file
    mdfile = mdutils.MdUtils(file_name=destination_filename)

    # Add headers
    mdfile.write(HEADERS["header"] + '\n')
    mdfile.write(HEADERS["sub_header"] + '\n')

    # Add questions (and hint or answers if required)
    for n in range(1, 101):
        mdfile.new_header(title=f"{n}. {QHA[f'q{n}']}", level=4, add_table_of_contents="n")
        if with_hints:
            mdfile.write(f"`{QHA[f'h{n}']}`")
        if with_solutions:
            mdfile.insert_code(QHA[f'a{n}'], language='python')

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
