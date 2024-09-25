
"""This module contains routines for building era-wise histograms."""

# Author: Nicolas Hug
# Author: Tim DeLise

cimport cython
from cython.parallel import prange

import numpy as np

from .common import HISTOGRAM_DTYPE
from .common cimport hist_struct
from .common cimport X_BINNED_DTYPE_C
from .common cimport G_H_DTYPE_C


# Notes:
# - IN views are read-only, OUT views are write-only
# - In a lot of functions here, we pass feature_idx and the whole 2d
#   histograms arrays instead of just histograms[feature_idx]. This is because
#   Cython generated C code will have strange Python interactions (likely
#   related to the GIL release and the custom histogram dtype) when using 1d
#   histogram arrays that come from 2d arrays.
# - The for loops are un-wrapped, for example:
#
#   for i in range(n):
#       array[i] = i
#
#   will become
#
#   for i in range(n // 4):
#       array[i] = i
#       array[i + 1] = i + 1
#       array[i + 2] = i + 2
#       array[i + 3] = i + 3
#
#   This is to hint gcc that it can auto-vectorize these 4 operations and
#   perform them all at once.


@cython.final
cdef class EraHistogramBuilder:
    """A Era Histogram builder... used to build histograms.

    A histogram is an array with n_bins entries of type HISTOGRAM_DTYPE. Each
    feature has its own histogram. A histogram contains the sum of gradients
    and hessians of all the samples belonging to each bin.

    There are different ways to build a histogram:
    - by subtraction: hist(child) = hist(parent) - hist(sibling)
    - from scratch. In this case we have routines that update the hessians
      or not (not useful when hessians are constant for some losses e.g.
      least squares). Also, there's a special case for the root which
      contains all the samples, leading to some possible optimizations.
      Overall all the implementations look the same, and are optimized for
      cache hit.

    Parameters
    ----------
    X_binned : ndarray of int, shape (n_samples, n_features)
        The binned input samples. Must be Fortran-aligned.
    n_bins : int
        The total number of bins, including the bin for missing values. Used
        to define the shape of the histograms.
    gradients : ndarray, shape (n_samples,)
        The gradients of each training sample. Those are the gradients of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians : ndarray, shape (n_samples,)
        The hessians of each training sample. Those are the hessians of the
        loss w.r.t the predictions, evaluated at iteration i - 1.
    hessians_are_constant : bool
        Whether hessians are constant.
    eras: ndarray, shape (n_samples,)
        the indices of the eras. If the original eras are numbered like 101, 102, 103, etc
        then this variables contains the indices for these eras, like 0, 1, 2, etc.
    n_eras: int
        the total number of eras
    """
    cdef public:
        const X_BINNED_DTYPE_C [::1, :] X_binned
        unsigned int n_features
        unsigned int n_bins
        G_H_DTYPE_C [::1] gradients
        G_H_DTYPE_C [::1] hessians
        G_H_DTYPE_C [::1] ordered_gradients
        G_H_DTYPE_C [::1] ordered_hessians
        unsigned char hessians_are_constant
        int n_threads
        int [::1] eras
        int n_eras

    def __init__(
        self, 
        const X_BINNED_DTYPE_C [::1, :] X_binned,
        unsigned int n_bins, 
        G_H_DTYPE_C [::1] gradients,
        G_H_DTYPE_C [::1] hessians,
        unsigned char hessians_are_constant,
        int n_threads,
        int [::1] eras,
        int n_eras
    ):

        self.X_binned = X_binned
        self.n_features = X_binned.shape[1]
        # Note: all histograms will have <n_bins> bins, but some of the
        # bins may be unused if a feature has a small number of unique values.
        self.n_bins = n_bins
        self.gradients = gradients
        self.hessians = hessians
        # for root node, gradients and hessians are already ordered
        self.ordered_gradients = gradients.copy()
        self.ordered_hessians = hessians.copy()
        self.hessians_are_constant = hessians_are_constant
        self.n_threads = n_threads
        self.eras = eras
        self.n_eras = n_eras

    def compute_era_histograms_brute(
        EraHistogramBuilder self,
        const unsigned int [::1] sample_indices,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        """Compute the histograms of the node by scanning through all the data.

        For a given feature, the complexity is O(n_samples)

        Parameters
        ----------
        sample_indices : array of int, shape (n_samples_at_node,)
            The indices of the samples at the node to split.

        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape (n_features, n_bins)
            The computed histograms of the current node.
        """
        cdef:
            int n_samples
            int feature_idx
            int f_idx
            int i
            # need local views to avoid python interactions
            unsigned char hessians_are_constant = self.hessians_are_constant
            int n_features = self.n_features
            int n_allowed_features = self.n_features
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients
            G_H_DTYPE_C [::1] gradients = self.gradients
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians
            G_H_DTYPE_C [::1] hessians = self.hessians
            # Histograms will be initialized to zero later within a prange
            hist_struct [:, :, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins, self.n_eras),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        with nogil:
            n_samples = sample_indices.shape[0]

            # Populate ordered_gradients and ordered_hessians. (Already done
            # for root) Ordering the gradients and hessians helps to improve
            # cache hit.
            if sample_indices.shape[0] != gradients.shape[0]:
                if hessians_are_constant:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                else:
                    for i in prange(n_samples, schedule='static',
                                    num_threads=n_threads):
                        ordered_gradients[i] = gradients[sample_indices[i]]
                        ordered_hessians[i] = hessians[sample_indices[i]]

            # Compute histogram of each feature
            for f_idx in prange(
                n_allowed_features, schedule='static', num_threads=n_threads
            ):
                if has_interaction_cst:
                    feature_idx = allowed_features[f_idx]
                else:
                    feature_idx = f_idx

                self._compute_era_histogram_brute_single_feature(
                    feature_idx, 
                    sample_indices, 
                    histograms
                )

        return histograms

    cdef void _compute_era_histogram_brute_single_feature(
            EraHistogramBuilder self,
            const int feature_idx,
            const unsigned int [::1] sample_indices,  # IN
            hist_struct [:, :, ::1] histograms # OUT
        ) noexcept nogil:
        """Compute the histogram for a given feature."""

        cdef:
            unsigned int n_samples = sample_indices.shape[0]
            const X_BINNED_DTYPE_C [::1] X_binned = self.X_binned[:, feature_idx]
            unsigned int root_node = X_binned.shape[0] == n_samples
            G_H_DTYPE_C [::1] ordered_gradients = self.ordered_gradients[:n_samples]
            G_H_DTYPE_C [::1] ordered_hessians = self.ordered_hessians[:n_samples]
            unsigned char hessians_are_constant = self.hessians_are_constant
            unsigned int bin_idx = 0
            unsigned int era_idx = 0

        for era_idx in range(self.n_eras):
            for bin_idx in range(self.n_bins):
                histograms[feature_idx, bin_idx, era_idx].sum_gradients = 0.
                histograms[feature_idx, bin_idx, era_idx].sum_hessians = 0.
                histograms[feature_idx, bin_idx, era_idx].count = 0

        if root_node:
            if hessians_are_constant:
                _build_era_histogram_root_no_hessian(
                    feature_idx, 
                    X_binned,
                    self.eras,
                    ordered_gradients,
                    histograms
                )
            else:
                _build_era_histogram_root(
                    feature_idx, 
                    X_binned,
                    self.eras,
                    ordered_gradients, 
                    ordered_hessians,
                    histograms
                )
        else:
            if hessians_are_constant:
                _build_era_histogram_no_hessian(
                    feature_idx,
                    sample_indices,
                    X_binned,
                    self.eras,
                    ordered_gradients,
                    histograms
                )
            else:
                _build_era_histogram(
                    feature_idx, 
                    sample_indices,
                    X_binned,
                    self.eras,
                    ordered_gradients,
                    ordered_hessians,
                    histograms
                )

    def compute_era_histograms_subtraction(
        EraHistogramBuilder self,
        hist_struct [:, :, ::1] parent_histograms,        # IN
        hist_struct [:, :, ::1] sibling_histograms,       # IN
        const unsigned int [:] allowed_features=None,  # IN
    ):
        """Compute the histograms of the node using the subtraction trick.

        hist(parent) = hist(left_child) + hist(right_child)

        For a given feature, the complexity is O(n_bins). This is much more
        efficient than compute_histograms_brute, but it's only possible for one
        of the siblings.

        Parameters
        ----------
        parent_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the parent.
        sibling_histograms : ndarray of HISTOGRAM_DTYPE, \
                shape (n_features, n_bins)
            The histograms of the sibling.
        allowed_features : None or ndarray, dtype=np.uint32
            Indices of the features that are allowed by interaction constraints to be
            split.

        Returns
        -------
        histograms : ndarray of HISTOGRAM_DTYPE, shape(n_features, n_bins, n_eras)
            The computed histograms of the current node.
        """

        cdef:
            int feature_idx
            int f_idx
            int n_features = self.n_features
            int n_allowed_features = self.n_features
            hist_struct [:, :, ::1] histograms = np.empty(
                shape=(self.n_features, self.n_bins, self.n_eras),
                dtype=HISTOGRAM_DTYPE
            )
            bint has_interaction_cst = allowed_features is not None
            int n_threads = self.n_threads

        if has_interaction_cst:
            n_allowed_features = allowed_features.shape[0]

        # Compute histogram of each feature
        for f_idx in prange(n_allowed_features, schedule='static', nogil=True, num_threads=n_threads):
            if has_interaction_cst:
                feature_idx = allowed_features[f_idx]
            else:
                feature_idx = f_idx

            _subtract_era_histograms(
                feature_idx,
                self.n_bins,
                self.n_eras,
                parent_histograms,
                sibling_histograms,
                histograms,
            )
        return histograms


cpdef void _build_era_histogram_naive(
        const int feature_idx,
        unsigned int [:] sample_indices,  # IN
        X_BINNED_DTYPE_C [:] binned_feature,  # IN
        int [:] eras, # IN
        G_H_DTYPE_C [:] ordered_gradients,  # IN
        G_H_DTYPE_C [:] ordered_hessians,  # IN
        hist_struct [:, :, :] out # OUT
    ) noexcept nogil:
    """Build histogram in a naive way, without optimizing for cache hit.

    Used in tests to compare with the optimized version."""
    cdef:
        unsigned int i
        unsigned int n_samples = sample_indices.shape[0]
        unsigned int sample_idx
        unsigned int bin_idx
        unsigned int era_idx

    for i in range(n_samples):
        sample_idx = sample_indices[i]
        bin_idx = binned_feature[sample_idx]
        era_idx = eras[sample_idx]
        out[feature_idx, bin_idx, era_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx, era_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx, era_idx].count += 1


cpdef void _subtract_era_histograms(
        const int feature_idx,
        unsigned int n_bins,
        unsigned int n_eras,
        hist_struct [:, :, ::1] hist_a,  # IN
        hist_struct [:, :, ::1] hist_b,  # IN
        hist_struct [:, :, ::1] out   # OUT
    ) noexcept nogil:
    """compute (hist_a - hist_b) in out"""
    cdef:
        unsigned int i = 0
        unsigned int era_idx = 0
    
    for i in range(n_bins):
        for era_idx in range(n_eras):
            out[feature_idx, i, era_idx].sum_gradients = (
                hist_a[feature_idx, i, era_idx].sum_gradients -
                hist_b[feature_idx, i, era_idx].sum_gradients
            )
            out[feature_idx, i, era_idx].sum_hessians = (
                hist_a[feature_idx, i, era_idx].sum_hessians -
                hist_b[feature_idx, i, era_idx].sum_hessians
            )
            out[feature_idx, i, era_idx].count = (
                hist_a[feature_idx, i, era_idx].count -
                hist_b[feature_idx, i, era_idx].count
            )


cpdef void _build_era_histogram(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const int [::1] eras,
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        const G_H_DTYPE_C [::1] ordered_hessians,  # IN
        hist_struct [:, :, ::1] out # OUT
    ) noexcept nogil: 
    """Return histogram for a given feature."""
    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

        unsigned int era_0
        unsigned int era_1
        unsigned int era_2
        unsigned int era_3
        unsigned int era_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        era_0 = eras[ sample_indices[i] ]
        era_1 = eras[ sample_indices[i + 1] ]
        era_2 = eras[ sample_indices[i + 2] ]
        era_3 = eras[ sample_indices[i + 3] ]

        out[feature_idx, bin_0, era_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1, era_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2, era_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3, era_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0, era_0].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_1, era_1].sum_hessians += ordered_hessians[i + 1]
        out[feature_idx, bin_2, era_2].sum_hessians += ordered_hessians[i + 2]
        out[feature_idx, bin_3, era_3].sum_hessians += ordered_hessians[i + 3]

        out[feature_idx, bin_0, era_0].count += 1
        out[feature_idx, bin_1, era_1].count += 1
        out[feature_idx, bin_2, era_2].count += 1
        out[feature_idx, bin_3, era_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        era_idx = eras[ sample_indices[i] ]
        out[feature_idx, bin_idx, era_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx, era_idx].sum_hessians += ordered_hessians[i]
        out[feature_idx, bin_idx, era_idx].count += 1


cpdef void _build_era_histogram_no_hessian(
        const int feature_idx,
        const unsigned int [::1] sample_indices,  # IN
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const int [::1] eras, # IN
        const G_H_DTYPE_C [::1] ordered_gradients,  # IN
        hist_struct [:, :, ::1] out # OUT
    ) noexcept nogil:  
    """Return histogram for a given feature, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_node_samples = sample_indices.shape[0]
        unsigned int unrolled_upper = (n_node_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

        unsigned int era_0
        unsigned int era_1
        unsigned int era_2
        unsigned int era_3
        unsigned int era_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[sample_indices[i]]
        bin_1 = binned_feature[sample_indices[i + 1]]
        bin_2 = binned_feature[sample_indices[i + 2]]
        bin_3 = binned_feature[sample_indices[i + 3]]

        era_0 = eras[ sample_indices[i] ]
        era_1 = eras[ sample_indices[i + 1] ]
        era_2 = eras[ sample_indices[i + 2] ]
        era_3 = eras[ sample_indices[i + 3] ]

        out[feature_idx, bin_0, era_0].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_1, era_1].sum_gradients += ordered_gradients[i + 1]
        out[feature_idx, bin_2, era_2].sum_gradients += ordered_gradients[i + 2]
        out[feature_idx, bin_3, era_3].sum_gradients += ordered_gradients[i + 3]

        out[feature_idx, bin_0, era_0].count += 1
        out[feature_idx, bin_1, era_1].count += 1
        out[feature_idx, bin_2, era_2].count += 1
        out[feature_idx, bin_3, era_3].count += 1

    for i in range(unrolled_upper, n_node_samples):
        bin_idx = binned_feature[sample_indices[i]]
        era_idx = eras[ sample_indices[i] ]
        out[feature_idx, bin_idx, era_idx].sum_gradients += ordered_gradients[i]
        out[feature_idx, bin_idx, era_idx].count += 1


cpdef void _build_era_histogram_root(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const int [::1] eras, # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        const G_H_DTYPE_C [::1] all_hessians,  # IN
        hist_struct [:, :, ::1] out  # OUT
    ) noexcept nogil: 
    """Compute histogram of the root node.

    Unlike other nodes, the root node has to find the split among *all* the
    samples from the training set. binned_feature and all_gradients /
    all_hessians already have a consistent ordering.
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

        unsigned int era_0
        unsigned int era_1
        unsigned int era_2
        unsigned int era_3
        unsigned int era_idx

    for i in range(0, unrolled_upper, 4):

        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        era_0 = eras[ i ]
        era_1 = eras[ i + 1 ]
        era_2 = eras[ i + 2 ]
        era_3 = eras[ i + 3 ]

        out[feature_idx, bin_0, era_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1, era_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2, era_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3, era_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0, era_0].sum_hessians += all_hessians[i]
        out[feature_idx, bin_1, era_1].sum_hessians += all_hessians[i + 1]
        out[feature_idx, bin_2, era_2].sum_hessians += all_hessians[i + 2]
        out[feature_idx, bin_3, era_3].sum_hessians += all_hessians[i + 3]

        out[feature_idx, bin_0, era_0].count += 1
        out[feature_idx, bin_1, era_1].count += 1
        out[feature_idx, bin_2, era_2].count += 1
        out[feature_idx, bin_3, era_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        era_idx = eras[i]
        out[feature_idx, bin_idx, era_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx, era_idx].sum_hessians += all_hessians[i]
        out[feature_idx, bin_idx, era_idx].count += 1


cpdef void _build_era_histogram_root_no_hessian(
        const int feature_idx,
        const X_BINNED_DTYPE_C [::1] binned_feature,  # IN
        const int [::1] eras, # IN
        const G_H_DTYPE_C [::1] all_gradients,  # IN
        hist_struct [:, :, ::1] out # OUT
    ) noexcept nogil:
    """Compute histogram of the root node, not updating hessians.

    Used when the hessians of the loss are constant (typically LS loss).
    """

    cdef:
        unsigned int i = 0
        unsigned int n_samples = binned_feature.shape[0]
        unsigned int unrolled_upper = (n_samples // 4) * 4

        unsigned int bin_0
        unsigned int bin_1
        unsigned int bin_2
        unsigned int bin_3
        unsigned int bin_idx

        unsigned int era_0
        unsigned int era_1
        unsigned int era_2
        unsigned int era_3
        unsigned int era_idx

    for i in range(0, unrolled_upper, 4):
        bin_0 = binned_feature[i]
        bin_1 = binned_feature[i + 1]
        bin_2 = binned_feature[i + 2]
        bin_3 = binned_feature[i + 3]

        era_0 = eras[ i ]
        era_1 = eras[ i + 1 ]
        era_2 = eras[ i + 2 ]
        era_3 = eras[ i + 3 ]

        out[feature_idx, bin_0, era_0].sum_gradients += all_gradients[i]
        out[feature_idx, bin_1, era_1].sum_gradients += all_gradients[i + 1]
        out[feature_idx, bin_2, era_2].sum_gradients += all_gradients[i + 2]
        out[feature_idx, bin_3, era_3].sum_gradients += all_gradients[i + 3]

        out[feature_idx, bin_0, era_0].count += 1
        out[feature_idx, bin_1, era_1].count += 1
        out[feature_idx, bin_2, era_2].count += 1
        out[feature_idx, bin_3, era_3].count += 1

    for i in range(unrolled_upper, n_samples):
        bin_idx = binned_feature[i]
        era_idx = eras[i]
        out[feature_idx, bin_idx, era_idx].sum_gradients += all_gradients[i]
        out[feature_idx, bin_idx, era_idx].count += 1
