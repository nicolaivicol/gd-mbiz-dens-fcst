import os
import numpy as np
import logging
import optuna
from typing import List
from optuna.distributions import FloatDistribution
from scipy.optimize import lsq_linear
import cvxpy as cp

from tsfcst.utils_tsfcst import smape

log = logging.getLogger(os.path.basename(__file__))
optuna.logging.disable_propagation()
optuna.logging.disable_default_handler()


class WeightsFinder:
    """ This finds the best weights to apply to a set of models to attain the lowest smape """

    y_true: np.array = None
    y_preds: np.ndarray = None
    model_names: List[str] = None
    pen_sum_not_1: float = 1000
    pen_not_equal: float = 0.1

    @staticmethod
    def trial_params(trial: optuna.Trial) -> np.array:
        pars = [trial.suggest_float(name=name_, low=0, high=1, step=0.10) for name_ in WeightsFinder.model_names]
        return np.array(pars)

    @staticmethod
    def equal_weights():
        n = WeightsFinder.y_preds.shape[1]
        return np.repeat(np.round(1 / n, 4), n)

    @staticmethod
    def best_lsq_linear():
        try:
            res = lsq_linear(
                A=WeightsFinder.y_preds,
                b=WeightsFinder.y_true,
                bounds=(0, 1),
                verbose=0,
                tol=0.0001,
                max_iter=100
            )
            weights = res.x
            assert isinstance(weights, np.ndarray)
            assert all(weights >= 0)
            return np.round(weights, 4)
        except:
            return WeightsFinder.equal_weights()

    @staticmethod
    def best_cvxpy():
        try:
            X = WeightsFinder.y_preds
            y = WeightsFinder.y_true
            beta = cp.Variable(len(WeightsFinder.model_names))  # define coefs as optimization variables
            objective = cp.Minimize(cp.sum_squares(X @ beta - y))  # use least squares, smape is not convex
            constraints = [cp.sum(beta) == 1, beta >= 0.0]
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)
            weights = beta.value
            assert isinstance(weights, np.ndarray)
            assert all(weights >= 0)
            return np.round(weights, 4)
        except:
            return WeightsFinder.equal_weights()

    @staticmethod
    def trials_params_predefined():
        names = WeightsFinder.model_names

        # variations of weights as best guesses to help the optimizer to search
        corners = [(np.array(names) == name_) * 1.0 for name_ in names]
        equal_weights_ = WeightsFinder.equal_weights()
        best_lsq_linear_ = WeightsFinder.best_lsq_linear()
        best_cvxpy_ = WeightsFinder.best_cvxpy()
        avg_bests = (best_lsq_linear_ + best_cvxpy_) / 2
        weights_vars = corners + [equal_weights_, best_lsq_linear_, best_cvxpy_, avg_bests]

        trials = []
        for weights in weights_vars:
            trial = optuna.trial.create_trial(
                params=dict(zip(names, weights)),
                distributions={name_: FloatDistribution(low=0, high=1) for name_ in names},
                value=WeightsFinder.smape_penalized(weights=weights),
            )
            trials.append(trial)

        return trials

    @staticmethod
    def smape(weights: np.array):
        y_pred = np.dot(WeightsFinder.y_preds, np.array(weights))
        smape_ = smape(y_true=WeightsFinder.y_true, y_pred=y_pred)
        return smape_

    @staticmethod
    def smape_penalized(weights: np.array):
        # high penalty if sum of coefs different from 1
        sum_weights = np.sum(weights)
        penalty_sum_not_1 = WeightsFinder.pen_sum_not_1 * (sum_weights - 1) ** 2

        # return early without computing the exact smape
        if abs(sum_weights - 1) > 0.50:
            smape_approx = abs(sum_weights - 1) * 100
            return smape_approx + penalty_sum_not_1

        smape_ = WeightsFinder.smape(weights)

        # small penalty for corner coefs / preference for more equal weights:
        penalty_not_equal = WeightsFinder.pen_not_equal * np.sum(np.square(weights - 1 / len(weights)))

        return smape_ + penalty_sum_not_1 + penalty_not_equal

    @staticmethod
    def objective(trial: optuna.Trial):
        """
        Objective function to minimize.
        Find best weights for the ensemble of models.
        Score to minimize is SMAPE adjusted with penalties.
        """
        weights = WeightsFinder.trial_params(trial)
        return WeightsFinder.smape_penalized(weights)

    @staticmethod
    def find_best(n_trials=200):
        study = optuna.create_study(direction='minimize')
        study.add_trials(WeightsFinder.trials_params_predefined())  # add trials for corner cases
        study.optimize(WeightsFinder.objective, n_trials=n_trials)  # search the entire space

        weights = list(study.best_params.values())
        smape_ = WeightsFinder.smape(weights)

        res = {
            'smape': smape_,
            'weights': weights,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'study': study
        }

        return res
