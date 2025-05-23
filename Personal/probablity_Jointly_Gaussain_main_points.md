# Jointly Gaussian Random Variables — A Deep Dive

We explore what it means for random variables to be **jointly Gaussian**, examining their structure, properties, and applications with rigor and intuition.

---

## 1. Definition and Fundamental Structure

Let $\mathbf{X} = [X_1, X_2, \dots, X_n]^T$ be an $n$-dimensional random vector.

### Definition (Joint Gaussianity)

$\mathbf{X}$ is **jointly Gaussian** if **any linear combination** of its components is Gaussian:

$$
\forall \mathbf{a} \in \mathbb{R}^n, \quad \mathbf{a}^T \mathbf{X} \sim \mathcal{N}(\mu_a, \sigma^2_a)
$$

where:
- $\mu_a = \mathbf{a}^T \mu$
- $\sigma^2_a = \mathbf{a}^T \Sigma \mathbf{a}$

$\mu = \mathbb{E}[\mathbf{X}]$, $\Sigma = \mathrm{Cov}(\mathbf{X})$

### Probability Density Function (PDF)

If $\Sigma$ is positive definite:

$$
f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)\right)
$$

- $\mu$: mean vector  
- $\Sigma$: covariance matrix

---

## 2. Covariance Matrix: Properties and Intuition

$$
\Sigma = \mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^T]
$$

### Properties

- **Symmetric**: $\Sigma = \Sigma^T$  
- **Positive Semi-Definite**: $\mathbf{v}^T \Sigma \mathbf{v} \ge 0$

### Interpretation

- Diagonal: variances  
- Off-diagonal: covariances  
- Encodes shape, orientation, and spread of the distribution

---

## 3. Marginal Distributions

Let:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix}, \quad
\mu = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \quad
\Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}
$$

Then:

$$
\mathbf{X}_1 \sim \mathcal{N}(\mu_1, \Sigma_{11})
$$

Marginals of a multivariate normal are again Gaussian.

---

## 4. Linear Transformations

Let $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$ where $A$ is $m \times n$:

$$
\mathbf{X} \sim \mathcal{N}(\mu, \Sigma) \Rightarrow \mathbf{Y} \sim \mathcal{N}(A\mu + \mathbf{b}, A\Sigma A^T)
$$

### Derivation via Characteristic Function

$$
\phi_Y(\omega) = \exp\left(i \omega^T (A\mu + b) - \frac{1}{2} \omega^T A\Sigma A^T \omega \right)
$$

---

## 5. Uncorrelatedness vs Independence

### Key Result

If $\mathbf{X}$ is jointly Gaussian, then:

$$
\text{Uncorrelated} \iff \text{Independent}
$$
### Proof Sketch

If $\Sigma$ is diagonal, the PDF factorizes into the product of marginals → independence.

---

## 6. Conditional Distributions

Given:
$$
\mathbf{X} = \begin{bmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{bmatrix} \sim \mathcal{N}\left(
\begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix},
\begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}
\right)
$$
Then:
$$
\mathbf{X}_1 \mid \mathbf{X}_2 = x_2 \sim \mathcal{N}(\mu_{1|2}, \Sigma_{1|2})
$$
Where:

- $\mu_{1|2} = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2)$  
- $\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}$

---

## 7. Characteristic Function
$$
\phi_{\mathbf{X}}(\omega) = \mathbb{E}[e^{i \omega^T \mathbf{X}}]
= \exp\left(i \omega^T \mu - \frac{1}{2} \omega^T \Sigma \omega \right)
$$

### Uses

- Derive linear transformations  
- Analyze convergence (via Lévy's continuity theorem)  
- Confirm marginal and conditional structure

---

## 8. Applications

### Signal Processing

- Kalman filters (recursive estimation)  
- Optimal filtering (Wiener filters)

### Bayesian Inference

- Conjugate priors for linear Gaussian models  
- Analytical posteriors

### Machine Learning

- Gaussian Processes (infinite-dimensional generalization)  
- PCA (based on covariance eigendecomposition)

### Statistics

- Hypothesis testing: Hotelling’s $T^2$  
- MANOVA, LDA assume Gaussian classes

---

*End of Our Deep Dive.*