# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _criterion.pyx for implementation details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef DOUBLE_t* y                     # Values of y
    cdef SIZE_t y_stride                 # Stride in y (since n_outputs >= 1)
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil
    cdef double proxy_impurity_improvement(self) nogil

cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride

cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total

cdef class VarianceCriterion:
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef DOUBLE_t* y                     # Values of y
    cdef SIZE_t y_stride                 # Stride in y (since n_outputs >= 1)
    cdef DOUBLE_t* w                     # Values of w
    cdef SIZE_t w_stride                 # Stride in w (since n_outputs >= 1)
    cdef DOUBLE_t* sample_weight         # Sample weights
    cdef double sq_sum_total             # Maybe not necessary...

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end
    cdef SIZE_t* split_indices           # if == 0, use for criterion calculations.

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double left_treated_sum_y              # sum of outcomes in left branch, treated samples
    cdef double left_control_sum_y              # sum of outcomes in left branch, control samples
    cdef double right_treated_sum_y             # sum of outcomes in right branch, treated samples
    cdef double right_control_sum_y             # sum of outcomes in right branch, control samples

    cdef SIZE_t left_treated_n                  # number of treated samples in left branch
    cdef SIZE_t right_treated_n                 # number of treated samples in right branch
    cdef SIZE_t left_control_n                  # number of control samples in left branch
    cdef SIZE_t right_control_n                 # number of control samples in right branch

    cdef double treated_sum_y                   # Sum of y over all treated
    cdef double control_sum_y                   # Sum of y over all control
    cdef SIZE_t treated_n                       # Number of treated
    cdef SIZE_t control_n                       # Number of control

    cdef double left_tau                 # Estimate of treatment effect in left child
    cdef double right_tau                # Estimate of treatment effect in right child
    cdef double tau                      # Estimate of treatment effect in parent node

    cdef SIZE_t binary_outcome                  # 1 if the outcome is binary, 0 otherwise.

    cdef double* sum_total          # For classification criteria, the sum of the
    # weighted count of each label. For regression,
    # the sum of w*y. sum_total[k] is equal to
    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* w, SIZE_t w_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end, SIZE_t* split_indices) nogil
    cdef void reset(self) nogil
    cdef void reverse_reset(self) nogil
    cdef void update(self, SIZE_t new_pos) nogil

    cpdef void set_binary_outcome(self, SIZE_t new_value)
    cdef double continuous_outcome_objective_improvement(self, DOUBLE_t variance_tau, DOUBLE_t* sum_tau, DOUBLE_t* sum_tau_sq, SIZE_t total_n) nogil
    cdef double binary_outcome_objective_improvement(self, DOUBLE_t variance_tau, DOUBLE_t* sum_tau, DOUBLE_t* sum_tau_sq, SIZE_t total_n) nogil
    cdef double objective_improvement(self, DOUBLE_t variance_tau, DOUBLE_t* sum_tau, DOUBLE_t* sum_tau_sq, SIZE_t total_n) nogil
    cdef void node_value(self, double* dest) nogil
