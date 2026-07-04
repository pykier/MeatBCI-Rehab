import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fft_bandpass(X, srate, low, high):
    X = np.asarray(X, dtype=np.float64)
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1.0 / float(srate))
    spectrum = np.fft.rfft(X, axis=-1)
    mask = (frequencies >= float(low)) & (frequencies <= float(high))
    spectrum[..., ~mask] = 0
    return np.fft.irfft(spectrum, n=X.shape[-1], axis=-1)


def symmetrize(matrix):
    return (matrix + matrix.T) / 2.0


def matrix_product(left, right):
    """Small deterministic matrix product that avoids platform BLAS crashes."""
    return np.einsum("ij,jk->ik", left, right)


def symmetric_eigh(matrix, tolerance=1e-10, max_iterations=None):
    """Jacobi eigensolver for small symmetric EEG covariance matrices."""
    work = symmetrize(np.asarray(matrix, dtype=np.float64)).copy()
    size = work.shape[0]
    vectors = np.eye(size, dtype=np.float64)
    if max_iterations is None:
        max_iterations = max(32, 100 * size * size)

    for _ in range(max_iterations):
        upper = np.triu(np.abs(work), k=1)
        flat_index = int(np.argmax(upper))
        p, q = np.unravel_index(flat_index, upper.shape)
        if upper[p, q] < tolerance:
            break

        app = work[p, p]
        aqq = work[q, q]
        apq = work[p, q]
        angle = 0.5 * np.arctan2(2.0 * apq, aqq - app)
        cosine = float(np.cos(angle))
        sine = float(np.sin(angle))

        for index in range(size):
            if index in (p, q):
                continue
            aip = work[index, p]
            aiq = work[index, q]
            new_ip = cosine * aip - sine * aiq
            new_iq = sine * aip + cosine * aiq
            work[index, p] = work[p, index] = new_ip
            work[index, q] = work[q, index] = new_iq

        work[p, p] = (
            cosine * cosine * app
            - 2.0 * sine * cosine * apq
            + sine * sine * aqq
        )
        work[q, q] = (
            sine * sine * app
            + 2.0 * sine * cosine * apq
            + cosine * cosine * aqq
        )
        work[p, q] = work[q, p] = 0.0

        old_p = vectors[:, p].copy()
        old_q = vectors[:, q].copy()
        vectors[:, p] = cosine * old_p - sine * old_q
        vectors[:, q] = sine * old_p + cosine * old_q

    values = np.diag(work).copy()
    order = np.argsort(values)
    return values[order], vectors[:, order]


def spd_eigh(matrix, floor=1e-8):
    values, vectors = symmetric_eigh(matrix)
    return np.maximum(values, floor), vectors


def spd_log(matrix):
    values, vectors = spd_eigh(matrix)
    return matrix_product(vectors * np.log(values), vectors.T)


def spd_exp(matrix):
    values, vectors = symmetric_eigh(matrix)
    return matrix_product(vectors * np.exp(values), vectors.T)


def spd_invsqrt(matrix):
    values, vectors = spd_eigh(matrix)
    return matrix_product(vectors * (values ** -0.5), vectors.T)


def covariance_matrices(X, trace_normalize=False):
    X = np.asarray(X, dtype=np.float64)
    covs = []
    for trial in X:
        centered = trial - np.mean(trial, axis=-1, keepdims=True)
        covariance = np.einsum(
            "ik,jk->ij",
            centered,
            centered,
        ) / max(1, centered.shape[-1] - 1)
        ridge = max(float(np.trace(covariance)) / covariance.shape[0], 1.0) * 1e-6
        covariance = symmetrize(covariance) + ridge * np.eye(covariance.shape[0])
        if trace_normalize:
            covariance /= max(float(np.trace(covariance)), 1e-12)
        covs.append(covariance)
    return np.stack(covs)


def log_euclidean_mean(covariances):
    return spd_exp(np.mean([spd_log(covariance) for covariance in covariances], axis=0))


def upper_triangle_vector(matrix):
    indices = np.triu_indices_from(matrix)
    vector = matrix[indices].copy()
    off_diagonal = indices[0] != indices[1]
    vector[off_diagonal] *= np.sqrt(2.0)
    return vector


def tangent_features(covariances, reference):
    inverse_sqrt = spd_invsqrt(reference)
    return np.stack(
        [
            upper_triangle_vector(
                spd_log(
                    matrix_product(
                        matrix_product(inverse_sqrt, covariance),
                        inverse_sqrt,
                    )
                )
            )
            for covariance in covariances
        ]
    )


class BinaryCSP:
    def __init__(self, n_components=4):
        self.n_components = int(n_components)

    def fit(self, X, y):
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("BinaryCSP requires exactly two classes.")
        covs = covariance_matrices(X, trace_normalize=True)
        covariance_a = np.mean(covs[y == labels[0]], axis=0)
        covariance_b = np.mean(covs[y == labels[1]], axis=0)
        composite = covariance_a + covariance_b

        values, vectors = spd_eigh(composite)
        whitening = matrix_product(
            vectors * (values ** -0.5),
            vectors.T,
        )
        whitened_a = symmetrize(
            matrix_product(
                matrix_product(whitening, covariance_a),
                whitening,
            )
        )
        eigenvalues, eigenvectors = symmetric_eigh(whitened_a)
        order = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        filters = matrix_product(eigenvectors[:, order].T, whitening)
        self.filters_ = filters[: self.n_components]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        projected = np.einsum("fc,tcs->tfs", self.filters_, X)
        variances = np.var(projected, axis=-1)
        variances /= np.maximum(np.sum(variances, axis=1, keepdims=True), 1e-12)
        return np.log(np.maximum(variances, 1e-12))


class FBCSPSVMRM(BaseEstimator, ClassifierMixin):
    """Filter-bank CSP and Riemann tangent-space features classified by SVM."""

    def __init__(
        self,
        srate=250.0,
        bands=((8, 12), (12, 16), (16, 20), (20, 24), (24, 30)),
        n_csp_components=4,
        svm_c=1.0,
        svm_gamma="scale",
    ):
        self.srate = srate
        self.bands = bands
        self.n_csp_components = n_csp_components
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma

    def _prepare_band(self, X, band, fit=False):
        X_band = fft_bandpass(X, self.srate, band[0], band[1])
        X_band -= np.mean(X_band, axis=-1, keepdims=True)

        if fit:
            channel_mean = np.mean(X_band, axis=(0, 2), keepdims=True)
            channel_std = np.std(X_band, axis=(0, 2), keepdims=True)
            channel_std = np.maximum(channel_std, 1e-8)
            self.channel_stats_.append((channel_mean, channel_std))
        else:
            channel_mean, channel_std = self.channel_stats_[len(self._transform_stats_)]
            self._transform_stats_.append(True)

        return (X_band - channel_mean) / channel_std

    def _features(self, X, fit=False, y=None):
        features = []
        if fit:
            self.csps_ = []
            self.riemann_means_ = []
            self.channel_stats_ = []
        else:
            self._transform_stats_ = []

        for band_index, band in enumerate(self.bands):
            X_band = self._prepare_band(X, band, fit=fit)

            if fit:
                csp = BinaryCSP(n_components=self.n_csp_components)
                csp.fit(X_band, y)
                self.csps_.append(csp)

                covs = covariance_matrices(X_band)
                riemann_mean = log_euclidean_mean(covs)
                self.riemann_means_.append(riemann_mean)
            else:
                csp = self.csps_[band_index]
                covs = covariance_matrices(X_band)
                riemann_mean = self.riemann_means_[band_index]

            csp_features = csp.transform(X_band)
            riemann_features = tangent_features(covs, riemann_mean)
            features.extend((csp_features, riemann_features))

        return np.concatenate(features, axis=1)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("FBCSPSVMRM currently supports two MI classes.")

        features = self._features(X, fit=True, y=y)
        self.scaler_ = StandardScaler()
        features = self.scaler_.fit_transform(features)
        self.svm_ = SVC(
            kernel="rbf",
            C=self.svm_c,
            gamma=self.svm_gamma,
            probability=True,
            class_weight="balanced",
        )
        self.svm_.fit(features, y)
        self.n_features_in_ = X.shape[1]
        self.feature_count_ = features.shape[1]
        return self

    def transform(self, X):
        features = self._features(np.asarray(X, dtype=np.float64), fit=False)
        return self.scaler_.transform(features)

    def predict(self, X):
        return self.svm_.predict(self.transform(X))

    def predict_proba(self, X):
        return self.svm_.predict_proba(self.transform(X))
