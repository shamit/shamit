"""
Test functions
"""
__author__ = "Oliver Contier"


from ..shambrain import get_signal_spec


def test_get_signal_spec():

    spec = [{'trial_type': 'some_condition',
             'onsets': [5, 50, 100],
             'durations': [1, 1, 1]},
            {'trial_type': 'some_condition',
             'onsets': [3, 30, 90],
             'durations': [2, 2, 2]}]

    get_signal_spec(infile, spec)


# def test_univ_neural_signal():
#     assert univ_neural_signal([], 2, 0) == []