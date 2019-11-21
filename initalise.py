import numpy as np

from questsions_dict import qha


def question(n):
    print(f'{n}. ' + qha[f'q{n}'])


def hint(n):
    print(qha[f'h{n}'])


def answer(n):
    print(qha[f'a{n}'])


def pick():
    n = np.random.randint(1, 100)
    question(n)
