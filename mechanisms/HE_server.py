import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.special import logsumexp



class HE_Computations:
    def __init__(self, domain, workload_domain_size, candidates, enc_noise_measure, enc_noise_select):
        self.domain = domain
        self.workload_domain_size = workload_domain_size
        self.candidates = candidates

        self.answers_encrypted = None
        self.used_up_guassian_samples = 0
        self.used_up_gumble_samples = 0
        self.enc_noise_measure = enc_noise_measure
        self.enc_noise_select = enc_noise_select

        # Pre-compute column indices for each attribute in OHE matrix
        self.attr_column_ranges = {}
        col_offset = 0
        for attr in domain.attrs:
            n_bins = domain[attr]
            self.attr_column_ranges[attr] = (col_offset, col_offset + n_bins)
            col_offset += n_bins

    # add chunking method for noise

    def compute(self, enc_data):
        """
        Compute marginals for all candidates using OHE data.

        Args:
            enc_data: One-hot encoded data matrix (n_records, total_bins)

        Returns:
            list: List of marginals where index corresponds to candidates list
        """
        answers_enc = []

        for cl in self.candidates:
            if len(cl) == 1:
                # One-way marginal: sum across columns
                marginal = self._compute_oneway(enc_data, cl[0])
            elif len(cl) == 2:
                # Two-way marginal: element-wise multiply and sum
                marginal = self._compute_twoway(enc_data, cl[0], cl[1])
            else:
                # K-way marginal
                marginal = self._compute_kway(enc_data, cl)

            answers_enc.append(marginal)
        self.answers_encrypted = answers_enc
        #return answers_enc

    def _compute_oneway(self, enc_data, attr):
        """
        Compute one-way marginal for a single attribute.
        Sum across columns corresponding to that attribute.
        """
        start_col, end_col = self.attr_column_ranges[attr]
        # Sum across columns: ω(N-1) additions
        marginal = np.sum(enc_data[:, start_col:end_col], axis=0)
        return marginal

    def _compute_twoway(self, enc_data, attr1, attr2):
        """
        Compute two-way marginal for two attributes.
        Element-wise multiply columns and sum.
        """
        start1, end1 = self.attr_column_ranges[attr1]
        start2, end2 = self.attr_column_ranges[attr2]

        # Get columns for each attribute
        cols1 = enc_data[:, start1:end1]  # shape: (N, ω1)
        cols2 = enc_data[:, start2:end2]  # shape: (N, ω2)

        # Create 2D marginal: (ω1 x ω2)N multiplications; (ω1 x ω2)(N-1) additions
        n_bins1 = end1 - start1
        n_bins2 = end2 - start2
        marginal = np.zeros((n_bins1, n_bins2))

        for i in range(n_bins1):
            for j in range(n_bins2):
                # Element-wise multiply and sum
                marginal[i, j] = np.sum(cols1[:, i] * cols2[:, j])

        return marginal.flatten()

    def _compute_kway(self, enc_data, clique):
        """
        Compute k-way marginal for k attributes.
        """
        # Get columns for each attribute in the clique
        columns = []
        shapes = []
        for attr in clique:
            start, end = self.attr_column_ranges[attr]
            columns.append(enc_data[:, start:end])
            shapes.append(end - start)

        # Create k-dimensional marginal
        marginal = np.zeros(shapes)

        # Generate all combinations of bin indices
        from itertools import product
        for indices in product(*[range(s) for s in shapes]):
            # Element-wise multiply across all attributes
            result = np.ones(enc_data.shape[0])
            for col_idx, bin_idx in enumerate(indices):
                result *= columns[col_idx][:, bin_idx]
            marginal[indices] = np.sum(result)

        return marginal.flatten()


    def measure(self, marginal_index, sigma):
        # we can add checks here on whether enough samples are available, if not waht to do, some logs and debug
        marginal = self.answers_encrypted[marginal_index]
        n_samples = len(marginal)
        noise = self.enc_noise_measure[self.used_up_guassian_samples : self.used_up_guassian_samples + n_samples]
        self.used_up_guassian_samples += n_samples
        y_enc = marginal + sigma * noise
        return y_enc

    def select_measure_worst_l1(self,candidates_indices, est_ans, epsilon, sigma, max_sensitivity,bias,wgt):
        errors = np.array([])
        # Select
        for marginal_index in candidates_indices.values():
            #reduce number of additions by taking only domain size
            bias_ = bias[marginal_index]
            wgt_ = wgt[marginal_index]
            x = self.answers_encrypted[marginal_index]
            xest = est_ans[marginal_index]
            # print("Error range:  -----> ",min(abs(x-xest)), max(abs(x-xest)))
            err = wgt_ * (np.linalg.norm(x - xest, 1) - bias_)
            noise = self.enc_noise_select[self.used_up_gumble_samples]
            self.used_up_gumble_samples += 1
            err = err + (2 * max_sensitivity / epsilon) * noise
            errors = np.append(errors, err)
        cl_dec = np.argmax(errors) # this is the index of the query in encrypted form

        # Measure
        marginal = self.answers_encrypted[cl_dec]
        n_samples = len(marginal)
        noise = self.enc_noise_measure[self.used_up_guassian_samples : self.used_up_guassian_samples + n_samples]
        self.used_up_guassian_samples += n_samples
        y_enc = marginal + sigma * noise
        cl = next((key for key, value in candidates_indices.items() if value == cl_dec), None)


        return cl, y_enc

    def select_measure_worst_squared_l2(self,candidates_indices, est_ans, epsilon, sigma, max_sensitivity,bias,wgt):
        errors = np.array([])
        # Select
        for marginal_index in candidates_indices.values():
            #reduce number of additions by taking only domain size
            bias_ = bias[marginal_index]
            wgt_ = wgt[marginal_index]
            x = self.answers_encrypted[marginal_index]
            xest = est_ans[marginal_index]
            # print("Error range:  -----> ",min(abs(x-xest)), max(abs(x-xest)))
            err = wgt_ * (np.sum((x-xest)**2) - bias_)
            noise = self.enc_noise_select[self.used_up_gumble_samples]
            self.used_up_gumble_samples += 1
            err = err + (2 * max_sensitivity / epsilon) * noise
            errors = np.append(errors, err)
        cl_dec = np.argmax(errors) # this is the index of the query in encrypted form

        # Measure
        marginal = self.answers_encrypted[cl_dec]
        n_samples = len(marginal)
        noise = self.enc_noise_measure[self.used_up_guassian_samples : self.used_up_guassian_samples + n_samples]
        self.used_up_guassian_samples += n_samples
        y_enc = marginal + sigma * noise
        cl = next((key for key, value in candidates_indices.items() if value == cl_dec), None)


        return cl, y_enc
