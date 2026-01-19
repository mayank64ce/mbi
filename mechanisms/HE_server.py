import numpy as np
import pandas as pd
import time
from scipy.special import softmax
from scipy.special import logsumexp
from openfhe import *
import openfhe_numpy as onp
from openfhe_numpy.tensor.ctarray import CTArray
from openfhe_numpy import ArrayEncodingType
from tqdm import tqdm

def check(vec_ct, keys):
    decrypted = []

    for ct in vec_ct:
        val = ct.decrypt(keys.secretKey, unpack_type="original")[0]
        decrypted.append(val)
    
    return np.array(decrypted)


def compute_squared_l2_norm_he(enc_error, len_vec):
    error_data = enc_error.data
    cc = error_data.GetCryptoContext()
    length = enc_error.original_shape[0]

    batch_size = cc.GetRingDimension() // 2
    
    norm = cc.EvalInnerProduct(error_data, error_data, length)

    norm = CTArray(
        norm, (len_vec, ), batch_size, (len_vec, 1), ArrayEncodingType.ROW_MAJOR
    )

    return norm


class HE_Computations:
    def __init__(self, domain, workload_domain_size, candidates, enc_noise_measure, enc_noise_select):
        self.domain = domain
        self.workload_domain_size = workload_domain_size
        self.candidates = candidates

        self.setup_crypto_context()

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
        
        self.get_rotation_keys()
        
    
    def get_rotation_keys(self):
        """
        Compute marginals for all candidates using OHE data.

        Args:
            enc_data: One-hot encoded data matrix (n_records, total_bins)

        Returns:
            list: List of marginals where index corresponds to candidates list
        """

        max_rot = 0
        print("Determining rotation keys...")
        for cl in tqdm(self.candidates):
            if len(cl) == 1:
                # One-way marginal: sum across columns
                start, end = self.attr_column_ranges[cl[0]]
                max_rot = max(max_rot, end-start)

            elif len(cl) == 2:
                # Two-way marginal: element-wise multiply and sum
                start1, end1 = self.attr_column_ranges[cl[0]]
                start2, end2 = self.attr_column_ranges[cl[1]]
                max_rot = max(max_rot, (end2-start2)*(end1-start1))

        onp.gen_rotation_keys(self.keys.secretKey, np.arange(0, max_rot+2).tolist())
                

    
    def combine(self, vecs):
        # how to handle if vecs[0] has no shape ??
        # batch_size = self.crypto_context.GetRingDimension() // 2
        sum_vec = onp.array(
            cc=self.crypto_context,
            data=np.zeros(self.len_vec),
            batch_size=self.batch_size,
            order=onp.ROW_MAJOR,
            mode="tile",
            fhe_type="C",
            public_key=self.keys.publicKey,
        )

        for i in range(len(vecs)):
            # import pdb; pdb.set_trace()
            vecs[i].shape = self.pt_select.shape
            try:
                sum_vec = sum_vec + onp.roll(self.pt_select * vecs[i], i)
            except:
                import pdb; pdb.set_trace()

        return sum_vec

    # add chunking method for noise
    def setup_crypto_context(self):
        """
        Here we define the CKKS FHE parameters and define the crypto context for encoding, encryption, decryption and decoding
        We set the crypto_context and keys here
        """
        mult_depth = 10

        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(mult_depth)
        params.SetScalingModSize(59)
        params.SetFirstModSize(60)
        params.SetScalingTechnique(FIXEDAUTO)
        params.SetKeySwitchTechnique(HYBRID)
        params.SetSecretKeyDist(UNIFORM_TERNARY)

        cc = GenCryptoContext(params)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        keys = cc.KeyGen()

        cc.EvalMultKeyGen(keys.secretKey)
        cc.EvalSumKeyGen(keys.secretKey)

        self.crypto_context = cc
        self.keys = keys
        

        self.batch_size = cc.GetRingDimension() // 2

        self.len_vec = self.batch_size

        rots = []

        for i in np.arange(-33,34):
            # print(i)
            rots.append(i)

        
        # onp.gen_rotation_keys(keys.secretKey, rots)

        selector_vector = np.zeros(self.len_vec)
        selector_vector[0] = 1.0

        self.pt_select = onp.array(
            cc=cc,
            data=selector_vector,
            batch_size=self.batch_size,
            order=onp.ROW_MAJOR,
            mode="tile",
            fhe_type="P",
            public_key=keys.publicKey,
        )
    
    def encrypt_data(self, data):
        enc_data = []

        r, c = data.shape

        for j in range(c):
            col = data[:, j]
            ct_col = onp.array(
                cc=self.crypto_context,
                data=col,
                batch_size=self.batch_size,
                order=onp.ROW_MAJOR,
                mode="tile",
                fhe_type="C",
                public_key=self.keys.publicKey,
            )

            enc_data.append(ct_col)
        
        self.enc_data = enc_data
    
    def encrypt_noise(self, noise_vector, fhe_type="C"):
        enc_noise = []
        for val in tqdm(noise_vector):
            new_val = np.zeros(self.len_vec)
            new_val[0] = val
            ct_noise = onp.array(
                cc=self.crypto_context,
                data=new_val,
                batch_size=self.batch_size,
                order=onp.ROW_MAJOR,
                mode="tile",
                fhe_type=fhe_type,
                public_key=self.keys.publicKey,
            )
            enc_noise.append(ct_noise)

        return enc_noise

    def compute(self, enc_data):
        """
        Compute marginals for all candidates using OHE data.

        Args:
            enc_data: One-hot encoded data matrix (n_records, total_bins)

        Returns:
            list: List of marginals where index corresponds to candidates list
        """
        answers_enc = []
        print("Running compute step...")
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
    
    def compute_he(self, data):
        start = time.time()
        # encrypt the data
        self.encrypt_data(data)

        answers_enc = []
        # answers = []

        for cl in tqdm(self.candidates):
            if len(cl) == 1:
                # One-way marginal: sum across columns
                marginal_he = self._compute_oneway_he(self.enc_data, cl[0])
                # marginal = self._compute_oneway(data, cl[0])
            elif len(cl) == 2:
                # Two-way marginal: element-wise multiply and sum
                marginal_he = self._compute_twoway_he(self.enc_data, cl[0], cl[1])
                # marginal = self._compute_twoway(data, cl[0], cl[1])
            else:
                # K-way marginal
                continue
                marginal = self._compute_kway(data, cl)

            answers_enc.append(marginal_he)
            # answers.append(marginal)
        # breakpoint()
        self.answers_encrypted = answers_enc
        elapsed = time.time() - start

        print(f"Compute step ran for {elapsed / 60} mins")

        del self.enc_data

    def _compute_oneway(self, enc_data, attr):
        """
        Compute one-way marginal for a single attribute.
        Sum across columns corresponding to that attribute.
        """
        start_col, end_col = self.attr_column_ranges[attr]
        # Sum across columns: ω(N-1) additions
        marginal = np.sum(enc_data[:, start_col:end_col], axis=0)
        return marginal
    
    def _compute_oneway_he(self,enc_data, attr):
        start, end = self.attr_column_ranges[attr]

        ans = []

        for i in range(start, end):
            # compute sum of column i
            col_sum = enc_data[i].sum(axis=0) # this here is a single number
            # we need to make it a vector again.
            # so we need to make a new null vector and set its first value as this value
            col_sum = self.combine([col_sum])

            ans.append(col_sum)

        return ans



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
    
    def _compute_twoway_he(self, enc_data, attr1, attr2):
        start1, end1 = self.attr_column_ranges[attr1]
        start2, end2 = self.attr_column_ranges[attr2]

        ans = []
        for i in range(start1, end1):
            col_1 = enc_data[i]
            for j in range(start2, end2):
                col_2 = enc_data[j]
                # val = col_1 @ col_2
                val = (col_1 * col_2).sum(axis=0)
                val = self.combine([val])
                ans.append(val)

        return ans
            

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

    def measure_he(self, marginal_index, sigma):
        # we can add checks here on whether enough samples are available, if not waht to do, some logs and debug
        marginal = self.answers_encrypted[marginal_index]
        n_samples = len(marginal)

        noise = self.enc_noise_measure[self.used_up_guassian_samples : self.used_up_guassian_samples + n_samples]

        noise_enc = self.encrypt_noise(noise)

        noised_marginal = []

        for i, ct in enumerate(marginal):
            noised_marginal.append((ct + noise_enc[i] * float(sigma)).decrypt(self.keys.secretKey, unpack_type="original")[0])

        noised_marginal = np.array(noised_marginal)
        
        return noised_marginal

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

        epsilon = float(epsilon)

        # Select
        for marginal_index in candidates_indices.values():
            #reduce number of additions by taking only domain size
            bias_ = float(bias[marginal_index])
            wgt_ = float(wgt[marginal_index])
            x = self.answers_encrypted[marginal_index]
            xest_orig = est_ans[marginal_index]
            # print("Error range:  -----> ",min(abs(x-xest)), max(abs(x-xest)))

            # xest = np.clip(xest_orig, -1e10, 1e10)
            xest = np.where(np.abs(xest_orig) < 1e-10, 0.0, xest_orig)
            try:
                xest = self.encrypt_noise(xest, fhe_type="P")
            except:
                breakpoint()

            diff = []
            for a, b in zip(x, xest):
                diff.append(a-b)
            
            diff_comb = self.combine(diff)
            
            norm_he = compute_squared_l2_norm_he(diff_comb, self.len_vec)

            norm_he = (norm_he + (-bias_)) * wgt_

            # err = wgt_ * (np.sum((x-xest)**2) - bias_)
            noise = self.enc_noise_select[self.used_up_gumble_samples]

            noise_enc = self.encrypt_noise([noise])

            self.used_up_gumble_samples += 1
            err = norm_he + noise_enc[0] * (2 * max_sensitivity / epsilon)
            err = err.decrypt(self.keys.secretKey, unpack_type="original")[0]
            errors = np.append(errors, err)
        cl_dec = np.argmax(errors) # this is the index of the query in encrypted form
        # breakpoint()
        # Measure
        marginal = self.answers_encrypted[cl_dec]
        n_samples = len(marginal)
        noise = self.enc_noise_measure[self.used_up_guassian_samples : self.used_up_guassian_samples + n_samples]
        noise_enc = self.encrypt_noise(noise)
        self.used_up_guassian_samples += n_samples
        # y_enc = marginal + sigma * noise
        cl = next((key for key, value in candidates_indices.items() if value == cl_dec), None)

        noised_marginal = []

        for i, ct in enumerate(marginal):
            noised_marginal.append((ct + noise_enc[i] * float(sigma)).decrypt(self.keys.secretKey, unpack_type="original")[0])

        noised_marginal = np.array(noised_marginal)

        # breakpoint()
        return cl, noised_marginal
