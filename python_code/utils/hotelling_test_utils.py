import numpy as np


def run_hotelling_test(ht_mat, ht_s0, ht_s1, prev_ht_s0, prev_ht_s1, i, tx, user):
    pilot_number = tx[:, user].shape[0]
    if i is None:
        # Do for symbol 0
        ht_s0_pooled, n0_t_0 = calculate_pooled_cov_mat(ht_s0[user], prev_ht_s0[user])
        # Do for symbol 1
        ht_s1_pooled, n1_t_0 = calculate_pooled_cov_mat(ht_s1[user], prev_ht_s1[user])
        # combine
        ht_mat[user] = n0_t_0 / pilot_number * ht_s0_pooled + n1_t_0 / pilot_number * ht_s1_pooled
    else:
        # Do for symbol 0
        ht_s0_pooled, n0_t_0 = calculate_pooled_cov_mat(ht_s0[user][i], prev_ht_s0[user][i])
        # Do for symbol 1
        ht_s1_pooled, n1_t_0 = calculate_pooled_cov_mat(ht_s1[user][i], prev_ht_s1[user][i])
        # combine
        ht_mat[user][i] = n0_t_0 / pilot_number * ht_s0_pooled + n1_t_0 / pilot_number * ht_s1_pooled
    return ht_mat


def calculate_pooled_cov_mat(ht_s, prev_ht_s):
    n_t = np.shape(ht_s)[0]
    mu_t = np.sum(ht_s) / n_t
    cov_t = np.cov(ht_s)
    prev_n_t = np.shape(prev_ht_s)[0]
    prev_mu_t = np.sum(prev_ht_s) / prev_n_t
    prev_cov_t = np.cov(prev_ht_s)
    pooled_cov = ((n_t - 1) * cov_t + (prev_n_t - 1) * prev_cov_t) / (n_t + prev_n_t - 2)
    ht_s_pooled = (n_t * prev_n_t) / (n_t + prev_n_t) * \
                  np.transpose(mu_t - prev_mu_t) * \
                  np.transpose(pooled_cov) * (mu_t - prev_mu_t)
    return ht_s_pooled, n_t
