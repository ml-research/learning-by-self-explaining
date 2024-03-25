
import pdb
import sys


def run_experiments(dataset, args):

    if dataset == 'CUB':
        from CUB.train import (
            train_X_to_C,
            train_oracle_C_to_y_and_test_on_Chat,
            train_Chat_to_y_and_test_on_Chat,
            train_X_to_C_to_y,
            train_X_to_y
        )
        from CUB.train_src import train_Chat_to_y_and_test_on_Chat_LSX, faithfulness, sim, sim_clf
    else:
        print("This dataset is not implemented")
        exit(1)

    experiment = args[0].exp
    if experiment == 'Concept_XtoC':
        train_X_to_C(*args)

    elif experiment == 'Independent_CtoY':
        train_oracle_C_to_y_and_test_on_Chat(*args)

    elif experiment == 'Sequential_CtoY':
        train_Chat_to_y_and_test_on_Chat(*args)

    elif experiment == 'Joint':
        train_X_to_C_to_y(*args)

    elif experiment == 'Standard':
        train_X_to_y(*args)

    elif experiment == 'LSX':
        train_Chat_to_y_and_test_on_Chat_LSX(*args)

    elif experiment == 'faithfulness':
        faithfulness(*args)

    elif experiment == 'sim':
        sim(*args)

    elif experiment == 'sim_clf':
        sim_clf(*args)


def parse_arguments():
    # First arg must be dataset, and based on which dataset it is, we will parse arguments accordingly
    assert len(sys.argv) > 2, 'You need to specify dataset and experiment'
    from CUB.train import parse_arguments

    dataset = sys.argv[1].upper()
    experiment = sys.argv[2].upper()

    args = parse_arguments(experiment=experiment)
    return dataset, args


if __name__ == '__main__':

    dataset, args = parse_arguments()

    run_experiments(dataset, args)