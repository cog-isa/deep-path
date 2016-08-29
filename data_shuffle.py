import os
from random import randint


def split_data(path='data/sample/raw/'):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in onlyfiles:
        i = randint(0, 9)
        if i == 9:
            os.rename(os.path.join(path, f), os.path.join(path+'test/', f))
        elif i == 8:
            os.rename(os.path.join(path, f), os.path.join(path+'validate/', f))
        else:
            os.rename(os.path.join(path, f), os.path.join(path+'train/', f))


def join_data(path='data/sample/raw/'):
    onlyfiles = [f for f in os.listdir(path+'test/') if os.path.isfile(os.path.join(path+'test/', f))]
    for f in onlyfiles:
        os.rename(os.path.join(path+'test/', f), os.path.join(path, f))
    onlyfiles = [f for f in os.listdir(path+'validate/') if os.path.isfile(os.path.join(path+'validate/', f))]
    for f in onlyfiles:
        os.rename(os.path.join(path+'validate/', f), os.path.join(path, f))
    onlyfiles = [f for f in os.listdir(path+'train/') if os.path.isfile(os.path.join(path+'train/', f))]
    for f in onlyfiles:
        os.rename(os.path.join(path+'train/', f), os.path.join(path, f))


path = 'data/sample/raw/'
if not [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]:
    join_data(path)
split_data(path)
