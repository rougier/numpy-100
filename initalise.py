from questsions_dict import qha


def question(n):
    print(qha[f'q{n}'])


def hint(n):
    print(qha[f'h{n}'])


def answer(n):
    print(qha[f'a{n}'])
