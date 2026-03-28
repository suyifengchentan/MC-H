## 3. Methodology

### 3.1 Problem Formulation

Let $\mathcal{X} = \{x_i\}_{i=1}^{N}$ denote a dataset of $N$ unlabeled samples spanning potentially multiple granularity levels. Existing deep clustering methods implicitly assume a uniform feature granularity, applying a single clustering mechanism regardless of inter-class variance. However, this assumption fails in practice: fine-grained datasets (e.g., bird species, aircraft models) exhibit highly compact feature distributions where samples of different classes occupy overlapping, small spherical cap regions in representation space, whereas coarse-grained datasets (e.g., CIFAR-10) yield naturally dispersed features with high inter-class angular separation. The core challenge is therefore: how to design a unified mechanism that automatically senses local feature density and applies appropriately calibrated repulsive forces without manual granularity specification.

---

### 3.2 Hypersphere Representation Space

All feature vectors are L2-normalized and projected onto the unit hypersphere $S^{d-1} \subset \mathbb{R}^d$. Formally, for a raw feature $f_i \in \mathbb{R}^d$ extracted by a frozen CLIP encoder followed by a learnable Feature Adapter, the spherical feature is:

$$z_i = \frac{f_i}{\|f_i\|_2}, \quad z_i \in S^{d-1}$$

This hypersphere projection offers three advantages: (i) it eliminates magnitude interference and focuses on directional information; (ii) cosine similarity between features reduces to an inner product, $\cos(z_i, z_j) = \langle z_i, z_j \rangle$; and (iii) the hypersphere geometry provides a bounded, isotropic ambient space where angular distances are well-defined, facilitating both clustering compactness and diversity regularization.

---

### 3.3 DPP on the Hypersphere: Density-Adaptive Repulsion

#### 3.3.1 DPP Kernel Construction

We define the DPP kernel $L \in \mathbb{R}^{n \times n}$ over a mini-batch of $n$ samples as:

$$L = \Phi \Phi^\top, \quad \Phi_i = \tilde{q}_i \cdot z_i$$

where $z_i \in S^{d-1}$ is the hypersphere-projected feature, and $\tilde{q}_i$ is a normalized quality score derived from an initial non-negative network output $q_i$. If $q_i$ were left unconstrained, the model could trivially minimize the DPP objective $-\log\det(I+L)$ by scaling $L$ without geometric restructuring. To eliminate this absolute scale degree of freedom while preserving the model's ability to learn relative sample importance, we introduce a batch-wise log-centering parameterization (gauge fixing):

$$\tilde{q}_i = \exp\!\left(\log q_i - \frac{1}{n}\sum_{j=1}^n \log q_j\right)$$

This constraint strictly fixes the geometric mean of the batch qualities to 1 ($\prod_{i=1}^n \tilde{q}_i = 1$). Consequently, $\tilde{q}_i$ captures pure relative significance ($\tilde{q}_i/\tilde{q}_j = q_i/q_j$) without introducing meaningless global amplification, ensuring the optimization process is faithfully governed by geometric restructuring rather than scale drift. Each kernel entry thus decomposes as:

$$L_{ij} = \tilde{q}_i \tilde{q}_j \langle z_i, z_j \rangle$$

where the inner product $\langle z_i, z_j \rangle$ directly encodes pairwise cosine similarity. $L = \Phi\Phi^\top$ is PSD by construction, satisfying the standard DPP validity requirement.

#### 3.3.2 Optimization Perspective: Gradient and Repulsion Analysis

The core of our density-adaptive mechanism lies in the optimization of the log-determinant loss: $\mathcal{L}_{\mathrm{DPP}} = -\log\det(I_n + L)$. While the marginal kernel $K = L(L + I_n)^{-1}$ provides a probabilistic background for DPP sampling, the repulsion effect is best understood through the gradient of the loss function.

From an optimization perspective, the gradient of $\mathcal{L}_{\mathrm{DPP}}$ with respect to the kernel $L$ is:
$$\frac{\partial \mathcal{L}_{\mathrm{DPP}}}{\partial L} = -(I_n + L)^{-1}$$

This inverse covariance structure automatically concentrates the optimization force toward subspace directions that lack feature coverage, shifting features away from over-crowded regions:

- **Dense regime** (dominant eigenvalues $\mu_k \gg 1$, fine-grained): The few crowded directions correspond to large eigenvalues, which are heavily penalized by the inverse weighting $(1+\mu_k)^{-1}$, redirecting strong optimization forces into the abundant unoccupied subspaces.
- **Sparse regime** ($\mu_k \ll 1$, coarse-grained): Features are spread uniformly, resulting in a nearly isotropic gradient that exerts uniform pressure across all directions, producing negligible net geometric restructuring.

#### 3.3.3 Quantitative Relationship Between Local Density and Eigenspectrum

We establish the connection between batch-level feature density and DPP repulsion strength via the mean pairwise cosine similarity $\rho$ and its variance $\sigma^2$:

$$\rho = \frac{1}{n(n-1)} \sum_{i \neq j} \langle z_i, z_j \rangle, \quad \sigma^2 = \operatorname{Var}(\langle z_i, z_j \rangle)$$

Under the simplifying assumption that quality scores are approximately uniform ($\tilde{q}_i \approx 1$, consistent with the gauge-fixing constraint), the Frobenius norm of $L$ satisfies $\|L\|_F^2 = \sum_{i,j} L_{ij}^2 \approx n (1 + (n-1)(\rho^2 + \sigma^2))$. In fine-grained datasets, samples are highly compact (high $\rho$) and overlap unevenly (high $\sigma^2$), causing $\|L\|_F^2$ to expand significantly. 

**Under the empirically observed low effective rank condition of fine-grained feature matrices**, this expansion does not distribute uniformly; instead, it concentrates heavily on the top components, implying that the dominant eigenvalues ($\mu_1, \mu_2, \dots$) become extremely large. This yields the key chain:

$$\rho, \sigma^2 \uparrow \;\Longrightarrow\; \|L\|_F \uparrow \;\Longrightarrow\; \text{Dominant } \mu_k \uparrow \;\Longrightarrow\; \text{large unoccupied subspaces} \;\Longrightarrow\; \text{strong gradient force expanding into sparse regions (fine-grained)}$$

$$\rho, \sigma^2 \downarrow \;\Longrightarrow\; \|L\|_F \downarrow \;\Longrightarrow\; \mu_k \downarrow \;\Longrightarrow\; \text{uniform coverage} \;\Longrightarrow\; \text{isotropic gradient, negligible restructuring (coarse-grained)}$$

This demonstrates that the DPP on $S^{d-1}$ is inherently density-adaptive: no explicit granularity label is required.

---

### 3.4 Density-Aware Adaptive Loss Weighting

While the intrinsic eigenspectrum of the DPP naturally provides a geometry-aware optimization force (as shown in Section 3.3.3), we introduce an explicit adaptive diversity strength $\alpha_{\mathrm{div}}$ as an external amplifier. Instead of rigidly interpolating features—which compromises the hypersphere manifold—we directly modulate the overall weight of the DPP loss term to ensure the repulsion magnitude remains optimally scaled across extreme granularity variations.

The adaptive strength unifies data-driven density awareness with time-driven curriculum scheduling:

$$\alpha_{\mathrm{div}}^{(t)} = \gamma(t) \cdot \min\!\left(\alpha_{\mathrm{base}} \cdot \operatorname{sg}(\rho),\; 1\right)$$

where $\alpha_{\mathrm{base}}$ is a base hyperparameter, and $\gamma(t) \in [0,1]$ is a curriculum multiplier that monotonically increases with training epoch $t$. To prevent unstable training oscillations (a positive feedback loop where increased repulsion lowers $\rho$, subsequently dropping $\alpha_{\mathrm{div}}$ and causing re-collapse), we apply the stop-gradient operator $\operatorname{sg}(\cdot)$ to $\rho$. This ensures $\alpha_{\mathrm{div}}$ acts purely as an external modulation signal during backpropagation rather than a differentiable part of the computational graph.

---

### 3.5 Geometric Interpretation: Attraction–Repulsion Balance

The overall framework achieves a dynamic equilibrium by balancing cluster-forming attraction with DPP-driven global repulsion. The DPP term alone drives global subspace expansion:
$$\min_{\{z_i\}} \mathcal{L}_{\mathrm{DPP}} = -\log \det(I_d + \Phi^\top \Phi)$$

Since $L = \Phi\Phi^\top$ is positive semi-definite, $(I_n + \Phi\Phi^\top)$ is strictly positive definite. This decisively eliminates the $\det(L)=0$ degeneracy, ensuring stable optimization. By Sylvester's determinant identity ($\det(I_n + \Phi\Phi^\top) = \det(I_d + \Phi^\top\Phi)$), the loss encourages both the expansion of the feature volume and a more uniform distribution of singular values, effectively maximizing the effective rank of the batch feature matrix. 

**Equilibrium** corresponds to the balance between local/cluster-level attraction (from $\mathcal{L}_{\mathrm{inst}}$ and $\mathcal{L}_{\mathrm{cluster}}$) and the global repulsion of $\mathcal{L}_{\mathrm{DPP}}$. In this state, features form tight spherical caps per class with approximately uniformly distributed cap centers, analogous to the Thomson problem where discrete charges reach a stable minimum-energy configuration on $S^{d-1}$.

---

### 3.6 Multi-Granularity Unified Loss Function

The total training objective integrates complementary terms with clear spatial division of labor:

$$\mathcal{L} = \mathcal{L}_{\mathrm{inst}} + \lambda_1 \mathcal{L}_{\mathrm{cluster}} + \lambda_2 \mathcal{L}_{\mathrm{sub}} + \alpha_{\mathrm{div}} \mathcal{L}_{\mathrm{DPP}}$$

| Term | Geometric Role on $S^{d-1}$ |
|---|---|
| $\mathcal{L}_{\mathrm{inst}}$ | Local Pairwise-repulsion: symmetric push against individual negative samples |
| $\mathcal{L}_{\mathrm{cluster}}$ | Cluster-level attraction: contrastive learning between cluster centroids |
| $\mathcal{L}_{\mathrm{sub}}$ | Subspace Regularization: align cluster covariance with trusted anchor subspace |
| $\mathcal{L}_{\mathrm{DPP}}$ | Global Subspace-expansion: maximize spatial volume and effective rank |

**Subspace Regularization ($\mathcal{L}_{\mathrm{sub}}$)**: To regularize cluster covariance to be well-aligned with a trusted anchor subspace and suppress spurious variance leakage into noisy directions, we penalize the residual variance outside a "trusted" anchor subspace. For each cluster $k$, let $\Sigma_k$ be the weighted feature covariance and $V_k \in \mathbb{R}^{d \times r}$ be the orthonormal basis of the $r$-rank anchor subspace (derived from teacher features or stable historical representations). The loss is defined as:
$$\mathcal{L}_{\mathrm{sub}} = \frac{1}{K} \sum_{k=1}^K \|\Sigma_k - V_k V_k^\top \Sigma_k V_k V_k^\top\|_F$$
This ensures that cluster features are well-aligned with their respective anchor subspaces, regularizing the covariance structure to focus on meaningful discriminative dimensions.

By replacing the empirical uniformity loss with the theoretically grounded $\mathcal{L}_{\mathrm{DPP}}$, our framework elegantly unifies representation learning without redundant repulsions. $\mathcal{L}_{\mathrm{inst}}$ operates at a local, pairwise level, while $\mathcal{L}_{\mathrm{DPP}}$ acts globally to maximize the total spatial volume of the batch. This allows $\mathcal{L}_{\mathrm{DPP}}$ to efficiently penalize higher-order redundancies that standard contrastive loss cannot perceive.

---

### 3.7 Training Strategy

Curriculum scheduling. Regulated by the epoch-dependent multiplier $\gamma(t)$, the effective DPP strength $\alpha_{\mathrm{div}}^{(t)}$ starts at near zero in early epochs. This allows the model to first establish basic cluster structures using contrastive forces. $\gamma(t)$ then gradually increases, introducing the density-aware DPP repulsion to progressively widen inter-class separations without disrupting initial convergence.

Temperature annealing. The softmax temperature $\tau$ of the cluster projector is annealed from high to low, coordinated with the DPP strength schedule: high $\tau$ encourages exploration in early training; low $\tau$ consolidates sharp assignments in late training.

DPP application frequency. Following the existing `dpp_every_n_steps` mechanism, the DPP projection frequency increases dynamically as training converges, concentrating computational cost where it yields the greatest benefit.