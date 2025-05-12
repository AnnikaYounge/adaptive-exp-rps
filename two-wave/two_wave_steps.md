# Two-wave RPS implementation steps for simulation

Plan for implementation of the two-wave RPS algorithm. First constructing simulation. Best if modular so can extend to multi-wave and other.

### (Done) Step 1: Problem setup for simulation

_See lattice.py and underlying_causal.py for completed code_

* **Input**: Parameters $R$, $M$, $H$
* **Output**: Treatment lattice $\Lambda$ of size $R^M$
* **Tasks**:

  * Generate the lattice
  * Define a mechanistic causal function
  * Compute true treatment effects $\beta_v = \phi(v)$

### Step 2: First-wave allocation

* **Goal**: Allocate $n_1$ samples proportionally to boundary likelihood
* **Equation**:

  $ P(v \in \partial \Pi | H) = 1 - \prod_{i=1}^M \left(1 - \frac{2 \min(v_i, R - 1 - v_i)}{R - 1} \right)^{H - 1}$
* **Tasks**:

  * Compute $P(v \in \partial \Pi | H)$ for all $v \in \Lambda$
  * Normalize to get allocation $n_1(v)$ over lattice

### Step 3: First-wave simulation

* **Goal**: Generate first-wave observations
* **Tasks**:

  * Use $n_1(v)$ to simulate $y \sim \mathcal{N}(\beta_v, \sigma^2)$
  * Construct design matrix $D$ and outcome vector $y$

### Step 4: Construct RPS

* **Goal**: Enumerate Rashomon set based on wave 1 data
* **Tasks**:

  * Enumerate permissible partitions $\Pi \in \mathcal{P}^*_H$ using external RPS code
  * Compute posterior assuming uniform ℓ₀ prior over permissible partitions.
  * Prune to $\text{RPS}_1 = { \Pi : P(\Pi \mid Z_1) \geq (1 - \eta) P(\Pi{\text{MAP}} \mid Z_1) }$

### Step 5: Second-wave allocation

* **Goal**: Discriminate among $\text{RPS}_1$
* **Approach**:

  * Allocate $n_2$ to vertices that differ across $\Pi \in \text{RPS}_1$
### Step 6: Second-wave simulation and estimation

* **Tasks**:

  * Simulate $y$ from $n_2$ observations
  * Combine with first-wave data
  * Recompute posterior over $\text{RPS}_1$ (or just MAP)
  * Estimate $\hat{\tau}(v)$ for each $v$ using pooled data
  * Output $\hat{v} = \arg\max_v \hat{\tau}(v)$

### Step 7: Evaluation
  * True best $v^*$, estimated best $\hat{v}$
  * Regret: $\tau(v^*) - \tau(\hat{v})$ / Expected
  * RPS size, runtime, MAP posterior