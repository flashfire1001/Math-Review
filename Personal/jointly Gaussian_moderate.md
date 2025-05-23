# Jointly Gaussian Random Variables

This document provides a comprehensive explanation of jointly Gaussian random variables, covering their definition, core components, key properties, and some proof insights, structured into several parts.

---

## Part 1: The Definition and Core Components

A set of $n$ random variables $\mathbf{X} = [X_1, X_2, \dots, X_n]^T$ is defined as jointly Gaussian (or multivariate normal) if its joint probability distribution is specified by a particular probability density function (PDF).

The standard PDF for a non-singular jointly Gaussian distribution is:

$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\mathbf{\Sigma})}} \exp\left(-\frac{1}{2}(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})\right)$$

This formula applies when the covariance matrix $\mathbf{\Sigma}$ is **positive definite (PD)**, which ensures $\det(\mathbf{\Sigma}) > 0$ and the existence of $\mathbf{\Sigma}^{-1}$.

Let's revisit the components with added detail:

* **$\mathbf{x} \in \mathbb{R}^n$:** A vector representing a specific point in the $n$-dimensional space of possible outcomes. The PDF gives the probability density at this specific point.
* **$\mathbf{\mu} = E[\mathbf{X}] \in \mathbb{R}^n$:** The **mean vector**. $\mu_i = E[X_i]$ is the expected value of $X_i$. It determines the center of the distribution and the location of the PDF's peak.
* **$\mathbf{\Sigma} = E[(\mathbf{X} - \mathbf{\mu})(\mathbf{X} - \mathbf{\mu})^T] \in \mathbb{R}^{n \times n}$:** The **covariance matrix**. This is a symmetric matrix:
    * Diagonal elements $\Sigma_{ii} = Var(X_i)$ (variance of $X_i$).
    * Off-diagonal elements $\Sigma_{ij} = Cov(X_i, X_j)$ for $i \neq j$ (covariance between $X_i$ and $X_j$).
    The covariance matrix dictates the shape, orientation, and scale of the distribution's "bell."

* **$\det(\mathbf{\Sigma})$:** The determinant of the covariance matrix. For the PDF to be well-defined by the standard formula, $\det(\mathbf{\Sigma})$ must be positive ($\mathbf{\Sigma}$ is PD).
* **$\mathbf{\Sigma}^{-1}$:** The inverse of the covariance matrix (exists if $\mathbf{\Sigma}$ is PD), also called the **precision matrix**.
* **$(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu})$:** The squared Mahalanobis distance. This term quantifies how far $\mathbf{x}$ is from the mean $\mathbf{\mu}$, adjusted for the variables' variances and correlations. Points with the same Mahalanobis distance lie on ellipsoids centered at $\mathbf{\mu}$, which are the contours of equal probability density.

We often use the notation $\mathbf{X} \sim N(\mathbf{\mu}, \mathbf{\Sigma})$ to say that the vector $\mathbf{X}$ is jointly Gaussian with mean vector $\mathbf{\mu}$ and covariance matrix $\mathbf{\Sigma}$.

**The Singular Case (PSD but not PD):** If $\mathbf{\Sigma}$ is positive semi-definite (PSD) but *not* positive definite (i.e., $\det(\mathbf{\Sigma}) = 0$), the variables are linearly dependent. The distribution is concentrated on a linear subspace of $\mathbb{R}^n$. In this case, the distribution does not have a density with respect to the standard Lebesgue measure on $\mathbb{R}^n$ (the density is zero everywhere except on the subspace, where the standard formula breaks down). However, it is still considered a Gaussian distribution in a more general sense, and its properties are often analyzed using tools like characteristic functions or by working within the subspace. The standard PDF formula only applies to the non-singular case ($\Sigma$ is PD).

---

## Part 2: Properties of the Covariance Matrix ($\mathbf{\Sigma}$)

For any random vector $\mathbf{X}$, its covariance matrix $\mathbf{\Sigma} = E[(\mathbf{X} - \mathbf{\mu})(\mathbf{X} - \mathbf{\mu})^T]$ is always:

1.  **Symmetric:** $\mathbf{\Sigma} = \mathbf{\Sigma}^T$. This is a direct consequence of $Cov(X_i, X_j) = Cov(X_j, X_i)$.
2.  **Positive Semi-Definite (PSD):** For any non-zero real vector $\mathbf{v} \in \mathbb{R}^n$, it must be true that $\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} \ge 0$.
    * **Why?** Consider a linear combination of the variables $Y = \mathbf{v}^T \mathbf{X}$. The variance of $Y$ must be non-negative: $Var(Y) = Var(\mathbf{v}^T \mathbf{X}) = \mathbf{v}^T Cov(\mathbf{X}) \mathbf{v} = \mathbf{v}^T \mathbf{\Sigma} \mathbf{v} \ge 0$. Therefore, the covariance matrix must be PSD.
    * If $\mathbf{\Sigma}$ is **Positive Definite (PD)**, then $\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} > 0$ for all *non-zero* $\mathbf{v}$. This implies $\det(\mathbf{\Sigma}) > 0$ and the variables are not linearly dependent. This is the assumption for the standard PDF formula in Part 1.
    * If $\mathbf{\Sigma}$ is only **PSD** (not PD), then $\mathbf{v}^T \mathbf{\Sigma} \mathbf{v} = 0$ for at least one non-zero $\mathbf{v}$. This implies $\det(\mathbf{\Sigma}) = 0$ and the variables are linearly dependent.

For jointly Gaussian variables, the mean vector $\mathbf{\mu}$ and any valid PSD matrix $\mathbf{\Sigma}$ *completely define* the distribution.

---

## Part 3: Marginal Distributions

A fundamental property is that jointly Gaussian implies marginal Gaussianity.

**Property:** If $\mathbf{X} = [X_1, \dots, X_n]^T$ is jointly Gaussian $N(\mathbf{\mu}, \mathbf{\Sigma})$, then any sub-vector formed by selecting a subset of variables from $\mathbf{X}$ is also jointly Gaussian. Consequently, each individual variable $X_i$ is Gaussian (normal) with its mean $\mu_i$ and its variance $\Sigma_{ii}$: $X_i \sim N(\mu_i, \Sigma_{ii})$.

**Explanation:** If you have the $n$-dimensional joint Gaussian distribution, "looking" at the distribution of only a subset of variables (by integrating out the others in the joint PDF) results in a lower-dimensional distribution that is *also* Gaussian. For a single variable $X_i$, this marginal distribution is the familiar 1D bell curve determined by its specific mean and variance from the original $\mathbf{\mu}$ and $\mathbf{\Sigma}$.

**Proof Insight (via Characteristic Functions):** This property is cleanly shown using characteristic functions. If $\mathbf{X} \sim N(\mathbf{\mu}, \mathbf{\Sigma})$, its characteristic function is $\phi_{\mathbf{X}}(\mathbf{\omega}) = \exp\left(i\mathbf{\omega}^T \mathbf{\mu} - \frac{1}{2}\mathbf{\omega}^T \mathbf{\Sigma} \mathbf{\omega}\right)$. The characteristic function of a sub-vector is obtained by setting the corresponding $\omega_j$ values to zero. The resulting expression matches the characteristic function of a (jointly) Gaussian distribution with the appropriate sub-vector of $\mathbf{\mu}$ and the corresponding sub-matrix of $\mathbf{\Sigma}$. For a single variable $X_i$, setting all $\omega_j = 0$ except $\omega_i$ yields the characteristic function of $N(\mu_i, \Sigma_{ii})$.

**Important:** The converse is *not* true. Individual random variables $X_1, \dots, X_n$ can each be marginally Gaussian, but the vector $\mathbf{X} = [X_1, \dots, X_n]^T$ might *not* be jointly Gaussian. Their joint distribution could have a different structure.

---

## Part 4: Linear Transformations

Jointly Gaussian distributions are "closed" under linear transformations.

**Property:** If $\mathbf{X}$ is an $n$-dimensional jointly Gaussian vector $\mathbf{X} \sim N(\mathbf{\mu}, \mathbf{\Sigma})$, and $\mathbf{Y}$ is an $m$-dimensional vector defined by a linear transformation $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, where $\mathbf{A}$ is an $m \times n$ matrix of constants and $\mathbf{b}$ is an $m$-dimensional vector of constants, then $\mathbf{Y}$ is also a **jointly Gaussian** vector.

The parameters of the resulting Gaussian distribution are:
* Mean Vector: $\mathbf{\mu}_Y = E[\mathbf{Y}] = \mathbf{A}\mathbf{\mu} + \mathbf{b}$
* Covariance Matrix: $\mathbf{\Sigma}_Y = Cov(\mathbf{Y}) = \mathbf{A}\mathbf{\Sigma}\mathbf{A}^T$

**Explanation:** This property is fundamental because it means that applying operations like scaling, rotation, translation, or taking weighted sums to jointly Gaussian variables results in variables that are still jointly Gaussian. This simplifies analysis in many models.

**Proof (via Characteristic Functions):** This is the standard and most direct proof. The characteristic function of $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$ is:
$$\phi_{\mathbf{Y}}(\mathbf{v}) = E[e^{i\mathbf{v}^T \mathbf{Y}}] = E[e^{i\mathbf{v}^T (\mathbf{A}\mathbf{X} + \mathbf{b})}] = e^{i\mathbf{v}^T \mathbf{b}} E[e^{i(\mathbf{A}^T \mathbf{v})^T \mathbf{X}}]$$
Letting $\mathbf{\omega} = \mathbf{A}^T \mathbf{v}$, the expectation term is the characteristic function of $\mathbf{X}$ evaluated at $\mathbf{A}^T \mathbf{v}$:$$\phi_{\mathbf{Y}}(\mathbf{v}) = e^{i\mathbf{v}^T \mathbf{b}} \phi_{\mathbf{X}}(\mathbf{A}^T \mathbf{v})$$
Substitute the formula for $\phi_{\mathbf{X}}(\mathbf{\omega})$:
$$\phi_{\mathbf{Y}}(\mathbf{v}) = e^{i\mathbf{v}^T \mathbf{b}} \exp\left(i(\mathbf{A}^T \mathbf{v})^T \mathbf{\mu} - \frac{1}{2}(\mathbf{A}^T \mathbf{v})^T \mathbf{\Sigma} (\mathbf{A}^T \mathbf{v})\right)$$
Using properties of matrix transpose $(AB)^T = B^T A^T$:
 $(\mathbf{A}^T \mathbf{v})^T = \mathbf{v}^T \mathbf{A}$
$$= e^{i\mathbf{v}^T \mathbf{b}} \exp\left(i\mathbf{v}^T \mathbf{A} \mathbf{\mu} - \frac{1}{2}\mathbf{v}^T \mathbf{A} \mathbf{\Sigma} \mathbf{A}^T \mathbf{v}\right)$$
Combine the exponential terms:
$$= \exp\left(i\mathbf{v}^T \mathbf{b} + i\mathbf{v}^T (\mathbf{A} \mathbf{\mu}) - \frac{1}{2}\mathbf{v}^T (\mathbf{A} \mathbf{\Sigma} \mathbf{A}^T) \mathbf{v}\right)$$
Factor out $i\mathbf{v}^T$:
$$= \exp\left(i\mathbf{v}^T (\mathbf{A} \mathbf{\mu} + \mathbf{b}) - \frac{1}{2}\mathbf{v}^T (\mathbf{A} \mathbf{\Sigma} \mathbf{A}^T) \mathbf{v}\right)$$
This is the characteristic function of a Gaussian distribution with mean vector $\mathbf{A} \mathbf{\mu} + \mathbf{b}$ and covariance matrix $\mathbf{A} \mathbf{\Sigma} \mathbf{A}^T$, proving that $\mathbf{Y}$ is jointly Gaussian with these parameters.

---

## Part 5: Uncorrelatedness and Independence

This is a unique and critical property for jointly Gaussian variables.

**Property:** For jointly Gaussian random variables $\mathbf{X} = [X_1, \dots, X_n]^T$, they are pairwise uncorrelated if and only if they are mutually independent.
$$\mathbf{X} \sim N(\mathbf{\mu}, \mathbf{\Sigma}) \text{ and } \Sigma_{ij} = 0 \text{ for all } i \neq j \quad \iff \quad X_1, \dots, X_n \text{ are independent}$$
Being pairwise uncorrelated means the covariance matrix $\mathbf{\Sigma}$ is a diagonal matrix.

**Explanation:**
* Independence *always* implies uncorrelatedness for any random variables.
* The special property for jointly Gaussian variables is the reverse implication: if they are uncorrelated (their covariance matrix is diagonal), they *must* be independent. This is not true for general random variables.

**Proof (via PDF Factorization):**
If the variables are pairwise uncorrelated, the covariance matrix $\mathbf{\Sigma}$ is diagonal. Its inverse $\mathbf{\Sigma}^{-1}$ is also diagonal with entries $1/\Sigma_{ii}$, and $\det(\mathbf{\Sigma}) = \prod_{i=1}^n \Sigma_{ii}$.
Substitute these into the exponent of the joint PDF:
$$(\mathbf{x} - \mathbf{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{\mu}) = \sum_{i=1}^n \frac{(x_i - \mu_i)^2}{\Sigma_{ii}}$$Substitute this and the determinant into the joint PDF formula:$$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \prod_{i=1}^n \Sigma_{ii}}} \exp\left(-\frac{1}{2} \sum_{i=1}^n \frac{(x_i - \mu_i)^2}{\Sigma_{ii}}\right)$$
$$= \prod_{i=1}^n \frac{1}{\sqrt{2\pi \Sigma_{ii}}} \exp\left(-\frac{(x_i - \mu_i)^2}{2\Sigma_{ii}}\right) = \prod_{i=1}^n f_{X_i}(x_i)$$
Since the joint PDF factors into the product of the marginal PDFs, the variables $X_1, \dots, X_n$ are independent. The term $f_{X_i}(x_i)$ is the PDF of a $N(\mu_i, \Sigma_{ii})$ distribution.

---

## Part 6: Conditional Distributions

Knowing a subset of jointly Gaussian variables allows us to determine the distribution of the remaining variables, and that conditional distribution is also Gaussian.

**Property:** If a jointly Gaussian vector $\mathbf{X}$ is partitioned into $\mathbf{X}_1$ (size $k$) and $\mathbf{X}_2$ (size $n-k$), the conditional distribution of $\mathbf{X}_1$ given $\mathbf{X}_2 = \mathbf{x}_2$ is **jointly Gaussian**.

Using the partitioned mean vector $\mathbf{\mu} = \begin{bmatrix} \mathbf{\mu}_1 \\ \mathbf{\mu}_2 \end{bmatrix}$ and covariance matrix $\mathbf{\Sigma} = \begin{bmatrix} \mathbf{\Sigma}_{11} & \mathbf{\Sigma}_{12} \\ \mathbf{\Sigma}_{21} & \mathbf{\Sigma}_{22} \end{bmatrix}$ (assuming $\mathbf{\Sigma}_{22}$ is invertible, i.e., $X_2$ is non-singular), the conditional distribution $\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2$ is $N(\mathbf{\mu}_{1|2}, \mathbf{\Sigma}_{1|2})$ with:

* **Conditional Mean:** $E[\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2] = \mathbf{\mu}_1 + \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \mathbf{\mu}_2)$
    * This is the expected value of $$\mathbf{{X}_1}$$ given that we know $\mathbf{X}_2$ is $\mathbf{x}_2$. It's a linear function of the observed value $\mathbf{x}_2$.
* **Conditional Covariance:** $Cov(\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2) = \mathbf{\Sigma}_{11} - \mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}$
    * This matrix represents the remaining uncertainty in $\mathbf{{X}_1}$ after observing $\mathbf{X}_2$. The term $\mathbf{\Sigma}_{12}\mathbf{\Sigma}_{22}^{-1}\mathbf{\Sigma}_{21}$ represents the reduction in covariance due to the observation. This conditional covariance matrix is constant and does not depend on $\mathbf{x}_2$.

**Proof Insight:** The derivation analyzes the conditional PDF $f_{\mathbf{X}_1 | \mathbf{X}_2}(\mathbf{x}_1 | \mathbf{x}_2) = \frac{f_{\mathbf{X}}(\mathbf{x}_1, \mathbf{x}_2)}{f_{\mathbf{X}_2}(\mathbf{x}_2)}$. Substituting the Gaussian PDFs and manipulating the exponent of the ratio, particularly by completing the square with respect to $\mathbf{{x}_1}$, reveals that the conditional PDF has the form of a Gaussian PDF in $\mathbf{x}_1$ with the derived mean and covariance.

---

## Part 7: The Characteristic Function (A Powerful Tool)

The characteristic function offers an alternative and often simpler way to define and work with Gaussian distributions, especially for proving properties like linear transformations.

**Property:** A random vector $\mathbf{X}$ is jointly Gaussian $N(\mathbf{\mu}, \mathbf{\Sigma})$ if and only if its characteristic function is:

$$\phi_{\mathbf{X}}(\mathbf{\omega}) = E[e^{i\mathbf{\omega}^T \mathbf{X}}] = \exp\left(i\mathbf{\omega}^T \mathbf{\mu} - \frac{1}{2}\mathbf{\omega}^T \mathbf{\Sigma} \mathbf{\omega}\right)$$

for $\mathbf{\omega} \in \mathbb{R}^n$.

**Explanation:** The characteristic function uniquely defines the distribution. This compact formula encapsulates all the information about the Gaussian distribution. Its simple exponential-quadratic form in the exponent is key to its mathematical tractability. It's particularly useful for proving properties involving sums and linear transformations.

---

## Part 8: Where You See Them: Key Applications

The mathematical tractability and key properties of jointly Gaussian distributions make them ubiquitous models:

1.  **Signal Processing:** Often used to model random noise (e.g., thermal noise). Estimation techniques like the Kalman filter and Wiener filter are optimal for linear systems with Gaussian noise, fundamentally relying on the conditional distribution property.
2.  **Finance:** Asset returns are often modeled as jointly Gaussian for portfolio optimization (e.g., Markowitz portfolio theory) and risk analysis, although real-world returns often deviate (e.g., heavier tails).
3.  **Machine Learning:**
    * **Gaussian Processes (GPs):** A non-parametric model where function values at any finite set of points are treated as jointly Gaussian random variables. Used for flexible regression and classification with uncertainty estimates.
    * **Bayesian Inference:** When likelihoods and priors are Gaussian (or conjugate for Gaussian), posteriors are often Gaussian or related, simplifying Bayesian updates (e.g., in Bayesian Linear Regression).
    * **Factor Analysis & Probabilistic PCA:** Latent variable models assuming observed data are linear combinations of Gaussian factors plus Gaussian noise, leading to jointly Gaussian observed data.
    * **Gaussian Mixture Models (GMMs):** Used for clustering and density estimation; while a mixture is not necessarily Gaussian, the components are.

These applications leverage the properties discussed, particularly closure under linear operations (for modeling linear systems or applying linear transformations) and the simple form of conditional distributions (for estimation, prediction, and inference).# Jointly Gaussian Random Variables

This document provides a comprehensive explanation of jointly Gaussian random variables, covering their definition, core components, key properties, and some proof insights, structured into several parts.

---

