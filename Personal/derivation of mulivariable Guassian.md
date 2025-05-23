We’ll derive the PDF of an $n$-dimensional **multivariate normal distribution** $\mathbf{X} \sim \mathcal{N}(\mu, \Sigma)$, starting from the standard normal case and applying a **change of variables** via a linear transformation.

------

## 🔹 Step-by-Step Derivation of the Multivariate Normal PDF

### 1. **Start with the Standard Normal**

Let $\mathbf{Z} \in \mathbb{R}^n$ be a **standard multivariate normal**:

$\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I_n)$

Its PDF is:

$f_{\mathbf{Z}}(\mathbf{z}) = \frac{1}{(2\pi)^{n/2}} \exp\left(-\frac{1}{2} \mathbf{z}^T \mathbf{z} \right)$

------

### 2. **Linear Transformation to General Gaussian**

Let $\mathbf{X} = A\mathbf{Z} + \mu$, where: 

-   $A \in \mathbb{R}^{n \times n}$ is **invertible**
-   $\mu \in \mathbb{R}^n$ is the **mean**

Then:

-   $\mathbf{X} \sim \mathcal{N}(\mu, \Sigma)$
-   Where $\Sigma = AA^T$

(We'll later discuss how to choose such $A$.)

------

### 3. **Change of Variables Formula**

We apply the change of variables formula for PDFs:

Let $\mathbf{x} = A\mathbf{z} + \mu \Rightarrow \mathbf{z} = A^{-1}(\mathbf{x} - \mu)$

Then:

$f_{\mathbf{X}}(\mathbf{x}) = f_{\mathbf{Z}}(A^{-1}(\mathbf{x} - \mu)) \cdot \left| \det(A^{-1}) \right|$

Since $\det(A^{-1}) = 1 / \sqrt{|\Sigma|}$, we plug in:

------

### 4. **Plug in the Standard Normal PDF**

Recall:

$f_{\mathbf{Z}}(\mathbf{z}) = \frac{1}{(2\pi)^{n/2}} \exp\left(-\frac{1}{2} \mathbf{z}^T \mathbf{z} \right)$

So:

$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}} \exp\left( -\frac{1}{2} (A^{-1}(\mathbf{x} - \mu))^T (A^{-1}(\mathbf{x} - \mu)) \right) \cdot \frac{1}{|\det A|}$

Now simplify the quadratic form:

$(A^{-1}(\mathbf{x} - \mu))^T (A^{-1}(\mathbf{x} - \mu)) = (\mathbf{x} - \mu)^T (A^{-1})^T A^{-1} (\mathbf{x} - \mu)$

But:

$(A^{-1})^T A^{-1} = (AA^T)^{-1} = \Sigma^{-1}$

------

### 5. **Final Expression**

We now obtain the full multivariate Gaussian PDF:

$f_{\mathbf{X}}(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu) \right)$

------

## 🔹 Notes

-   The term $(\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)$ is the **Mahalanobis distance** squared.
-   The determinant $|\Sigma|$ adjusts for **volume scaling** in space — it’s the generalization of variance in 1D.

------

## 🔹 How to Choose $A$ such that $\Sigma = AA^T$?

There are multiple choices:

-   **Cholesky decomposition**: $A = \text{Chol}(\Sigma)$, where $A$ is lower-triangular.
-   **Spectral decomposition**: If $\Sigma = U \Lambda U^T$, then $A = U \Lambda^{1/2}$

Any such $A$ ensures $AA^T = \Sigma$.

------

Let me know if you want a diagram or animation showing how the transformation reshapes the standard Gaussian into an elliptical one, or if you’d like this derivation in LaTeX, PDF, or Markdown.