import json


def create_scores_file(out_file, **scores):
    with open(out_file, 'w') as f:
        json.dump(dict(_scores = scores),
                  f,
                  indent = 4)
