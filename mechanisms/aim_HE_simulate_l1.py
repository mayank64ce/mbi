"""Implementation of AIM: An Adaptive and Iterative Mechanism for DP Synthetic Data.

Note that with the default settings, AIM can take many hours to run.  You can configure
the runtime /utility tradeoff via the max_model_size flag.  We recommend setting it to 1.0
for debugging, but keeping the default value of 80 for any official comparisons to this mechanism.

Note that we assume in this file that the data has been appropriately preprocessed so that there are no large-cardinality categorical attributes.  If there are, we recommend using something like "compress_domain" from mst.py.  Since our paper evaluated already-preprocessed datastes, we did not implement that here for simplicity.
"""
import time

import joblib
import numpy as np
import itertools
from mbi import (
    Dataset,
    Domain,
    estimation,
    junction_tree,
    LinearMeasurement,
    LinearMeasurement,
)
from mechanism import Mechanism
from collections import defaultdict
from scipy.optimize import bisect
import pandas as pd
from mbi import Factor
import argparse
from HE_server import HE_Computations


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s) + 1)
    )


def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))


def compile_workload(workload):
    weights = {cl: wt for (cl, wt) in workload}
    workload_cliques = weights.keys()

    def score(cl):
        return sum(
            weights[workload_cl] * len(set(cl) & set(workload_cl))
            for workload_cl in workload_cliques
        )

    return {cl: score(cl) for cl in downward_closure(workload_cliques)}


def filter_candidates(candidates, model, size_limit):
    ans = {}
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = (
                junction_tree.hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        )
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans





class AIM(Mechanism):
    def __init__(
            self,
            epsilon,
            delta,
            prng=None,
            rounds=None,
            max_model_size=80,
            max_iters=1000,
            structural_zeros={},
    ):
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_iters = max_iters
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2 / np.pi) * sigma * model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt)

        max_sensitivity = max(
            sensitivity.values()
        )  # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def OHE(self, data):
        domain = data.domain
        df = data.df

        # Calculate total number of bins across all features
        total_bins = sum(domain[attr] for attr in domain.attrs)
        n_records = len(df)

        # Initialize OHE matrix
        ohe_matrix = np.zeros((n_records, total_bins), dtype=int)

        # Track the current column position in the OHE matrix
        col_offset = 0

        # Process each attribute in the domain
        for attr in domain.attrs:
            n_bins = domain[attr]  # number of bins for this attribute

            # Get the binned values for this attribute
            attr_values = df[attr].values

            # Set the appropriate bin to 1 for each record
            for row_idx, bin_val in enumerate(attr_values):
                # bin_val should be in range [0, n_bins-1]
                col_idx = col_offset + int(bin_val)
                ohe_matrix[row_idx, col_idx] = 1

            # Move offset to the next attribute's bins
            col_offset += n_bins

        return ohe_matrix

    def get_unit_guassian_samples(self, n):
        return self.gaussian_noise(1,n)

    def get_unit_gumbel_samples(self, n):
            return np.random.gumbel(loc=0, scale=1, size=n)

    def calculate_max_gaussian_samples(self, domain, workload, rho):
        pass
        return 10000

    def calculate_max_gumbel_samples(self, domain, workload, rho):
        pass
        return 10000



    def run(self, data, workload, num_synth_rows=None, initial_cliques=None):
        rounds = self.rounds or 16 * len(data.domain)
        candidates = compile_workload(workload)


        # Sikha start ----- COMPUTE
        domain = data.domain
        workload_domain_size = [domain.project(cl).size() for cl in candidates]
        max_domain_size = max(workload_domain_size)
        #answers = {cl: data.project(cl).datavector() for cl in candidates}
        ohe_data = self.OHE(data)
        data_enc = ohe_data.copy()

        gaussian_samples_needed = self.calculate_max_gaussian_samples(data.domain, workload, self.rho)
        gumbel_samples_needed = self.calculate_max_gumbel_samples(data.domain, workload, self.rho)


        enc_noise_measure = self.get_unit_guassian_samples(gaussian_samples_needed)
        enc_noise_select = self.get_unit_gumbel_samples(gumbel_samples_needed)
        he = HE_Computations(domain, workload_domain_size, candidates, enc_noise_measure, enc_noise_select)
        #answers_enc =
        he.compute(data_enc)
        # Sikha end


        sigma = np.sqrt(rounds / (2 * 0.9 * self.rho))
        epsilon = np.sqrt(8 * 0.1 * self.rho / rounds)

        if not initial_cliques:
            initial_cliques = [
                cl for cl in candidates if len(cl) == 1
            ]  # use one-way marginals
        oneway = [cl for cl in candidates if len(cl) == 1]

        measurements = []
        print("Initial Sigma", sigma)
        rho_used = len(oneway) * 0.5 / sigma**2
        # Sikha start ----- MEASURE ONE WAY
        oneway_indices = {value:i for i, value in enumerate(candidates) if value in oneway}
        #for cl in oneway_indices:
        for cl, marginal_index in oneway_indices.items():
            #marginal_index = oneway_indices[cl]
            y_enc = he.measure(marginal_index, sigma) # need to figure this out
            #x = data.project(cl).datavector()
            #y = x + self.gaussian_noise(sigma, x.size)
            y = y_enc.copy() # Decrypt here
            # Sikha end
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))

        zeros = self.structural_zeros
        # NOTE: Haven't incorproated structural zeros back yet after refactoring
        model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters, callback_fn=lambda *_: None
        )

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2 * (0.5 / sigma**2 + 1.0 / 8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2 * 0.9 * remaining))
                epsilon = np.sqrt(8 * 0.1 * remaining)
                terminate = True

            rho_used += 1.0 / 8 * epsilon**2 + 0.5 / sigma**2
            print('Budget Used', rho_used, '/', self.rho)
            size_limit = self.max_model_size * rho_used / self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)

            # Sikha start ----- SELECT and MEASURE
            small_candidates_indices = {value:i for i, value in enumerate(candidates) if value in small_candidates.keys()}
            est_ans = []
            for cl in candidates:
                data_vector = model.project(cl).datavector()
                #padded_data_vector = np.pad(data_vector, (0, max_domain_size - len(data_vector)), 'constant')
                est_ans.append(data_vector)

            bias = np.zeros(len(candidates))
            wgt = np.ones(len(candidates))
            sensitivity =[]
            for i,cl in zip(small_candidates_indices.values(),small_candidates.keys()):
                wt = small_candidates[cl]
                wgt[i] = wt
                bias[i] = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
                sensitivity.append(abs(wt))
            max_sensitivity = max(sensitivity)

            cl, y_enc = he.select_measure_worst_l1(small_candidates_indices, est_ans, epsilon, sigma, max_sensitivity,bias,wgt)
            y = y_enc.copy() # decrypt here

            # cl = self.worst_approximated(
            #     small_candidates, answers, model, epsilon, sigma
            # )
            # print('Measuring Clique', cl)
            n = data.domain.size(cl)
            # x = data.project(cl).datavector()
            # y = x + self.gaussian_noise(sigma, n)
            measurements.append(LinearMeasurement(y, cl, stddev=sigma))
            z = model.project(cl).datavector()
            # Sikha end

            # Warm start potentials from prior round
            # TODO: check if it helps to call maximal_subsets here
            pcliques = list(set(M.clique for M in measurements))
            potentials = model.potentials.expand(pcliques)
            model = estimation.mirror_descent(
                data.domain, measurements, iters=self.max_iters, potentials=potentials, callback_fn=lambda *_: None
            )
            w = model.project(cl).datavector()
            # print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            print("(!!!!!!!!!!!!!!!!!!!!!!)                    Error in this round", np.linalg.norm(w - z, 1))
            if np.linalg.norm(w - z, 1) <= sigma * np.sqrt(2 / np.pi) * n:
                print("(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma", sigma / 2)
                sigma /= 2
                epsilon *= 2

        print("Generating Data...")
        model = estimation.mirror_descent(
            data.domain, measurements, iters=self.max_iters, potentials=potentials
        )
        synth = model.synthetic_data(rows=num_synth_rows)

        return model, synth




def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/unosb_v1_clean_smallest.csv'
    # params['dataset'] = '../data/unosb_v1_clean.csv'
    params['domain'] = '../data/unosb_v1_smallest-domain.json'
    # params['domain'] = '../data/unosb_v1-domain.json'
    params["epsilon"] = 10
    params["delta"] = 1e-9
    params["noise"] = "laplace"
    params["max_model_size"] = 80
    params["max_iters"] = 1000
    params["degree"] = 2
    params["num_marginals"] = None
    params["max_cells"] = 10000
    params["save"] = '../data/synth_unosb_v1_he.csv'
    return params


if __name__ == "__main__":

    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument("--dataset", help="dataset to use")
    parser.add_argument("--domain", help="domain to use")
    parser.add_argument("--epsilon", type=float, help="privacy parameter")
    parser.add_argument("--delta", type=float, help="privacy parameter")
    parser.add_argument(
        "--max_model_size", type=float, help="maximum size (in megabytes) of model"
    )
    parser.add_argument("--max_iters", type=int, help="maximum number of iterations")
    parser.add_argument("--degree", type=int, help="degree of marginals in workload")
    parser.add_argument(
        "--num_marginals", type=int, help="number of marginals in workload"
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        help="maximum number of cells for marginals in workload",
    )
    parser.add_argument("--save", type=str, help="path to save synthetic data")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    # workload_all = list(itertools.combinations(data.domain, args.degree))
    # workload = list(itertools.combinations(data.domain, 1))
    #workload = workload + [("meld_bin", "event_status"), ("time_bin", "event_status"), ("meld_bin", "time_bin")]
    #workload = [("meld_bin", "event_status"), ("time_bin", "event_status"), ("meld_bin", "time_bin")]

    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        prng = np.random
        workload = [
            workload[i]
            for i in prng.choice(len(workload), args.num_marginals, replace=False)
        ]

    workload = [(cl, 1.0) for cl in workload]
    # workload_all = [(cl, 1.0) for cl in workload_all]
    start_time = time.time()
    mech = AIM(
        args.epsilon,
        args.delta,
        max_model_size=args.max_model_size,
        max_iters=args.max_iters,
    )
    model, synth = mech.run(data, workload)
    joblib.dump(model, "../data/aim_adult_generator_eps10.joblib")
    stop_time = time.time()
    print("Time taken to train and generate:", stop_time-start_time)
    # if args.save is not None:
    #     synth.df.to_csv(args.save, index=False)
    #
    #
    errors = []
    # for proj, wgt in workload_all:
    for proj, wgt in workload: #_all:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5 * wgt * np.linalg.norm(X / X.sum() - Y / Y.sum(), 1)
        errors.append(e)
    print("Average Error: ", np.mean(errors))