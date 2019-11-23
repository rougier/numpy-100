import numpy as np

import data_source as ds


def question(n):
    print(f'{n}. ' + ds.QHA[f'q{n}'])


def hint(n):
    print(ds.QHA[f'h{n}'])


def answer(n):
    print(ds.QHA[f'a{n}'])


def pick():
    n = np.random.randint(1, 100)
    question(n)
