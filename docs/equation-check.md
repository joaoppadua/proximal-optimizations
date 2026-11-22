# Equation check

Every equation the notebooks use, checked against its source paper (Task 10, 2026-11-21).
Each row gives the equation as the project uses it, the paper, where in the paper, and the
verdict. "Matches" allows notation differences that leave the math unchanged; the mapping is
noted. Equation numbers are those of the version read (listed in "Sources read" at the end).

Notation of this project: `x` prompt, `y` response, `y_t` token, `π_θ`, `π_ref`, `r`, `β`,
`A` advantage, `G` group size, `ε` clip range.

## Differences to resolve

1. **REINFORCE mean baseline includes the sample's own return** (`reinforce_loss`,
   `baseline="mean"`). Williams (1992) §4 and Theorem 2 (§5) require the baseline `b_ij` to be
   conditionally independent of the unit's output given `W` and `x`. The project's baseline is
   the batch mean of `R`, which contains `R_i` itself, so it is not independent of `y_i`. The
   consequence is exact and small: `E[(R_i − mean_j R_j) ∇log π(y_i)] = (1 − 1/B) · ∇J_β`
   (the other samples' returns are independent of `y_i` and `E[∇log π] = 0`). The expected
   update keeps its direction and is scaled by `1 − 1/B` (0.984 at B = 64), so it is still an
   ascent direction but is not, strictly, a REINFORCE algorithm in Williams' sense.
   *Which side is right:* both are correct statements; the code is not wrong. A notebook must not
   say that the batch-mean baseline is "unbiased" without the factor `1 − 1/B`, nor attribute it
   to Williams. A leave-one-out mean would satisfy Williams' condition exactly if wanted. No code
   was changed.

No other difference found between the project and the papers. Three paper-side points that do
not affect the project but that a notebook must not copy are listed under "Notes" below.

## Rows

| # | Equation as the project uses it | Paper | Where | Verdict |
|---|---|---|---|---|
| 1 | REINFORCE on `J_β`: `∇J_β ≈ mean_i[(R_i − b) ∇log π_θ(y_i\|x_i)]`, `R = r − β Σ_t (log π_θ(y_t) − log π_ref(y_t))` (detached), `∇log π_θ(y\|x) = Σ_t ∇log π_θ(y_t\|x, y_<t)` (`reinforce_loss`) | Williams 1992 | §4, update `Δw_ij = α_ij (r − b_ij) e_ij` (unnumbered) and Theorem 1; §5, episodic form eq. (11) `Δw_ij = α_ij (r − b_ij) Σ_t e_ij(t)` and Theorem 2 | Matches in form, with `e_ij = ∂ln g_i/∂w_ij` = score, the episodic sum over time steps = sum over tokens, `r` replaced by the regularised return `R`. The KL-regularised return is the project's own step, not Williams': it holds because `∇J_β = E[(r − β log(π_θ/π_ref)) ∇log π_θ] − β E[∇log π_θ]` and the last term is 0. Baseline: see Difference 1. |
| 2 | RLHF reward with the KL penalty: `R(x,y) = r(x,y) − β log(π(y\|x)/π_ref(y\|x))` | Ziegler et al. 2019 | §2, eq. (2) (`ρ` = `π_ref`) | Matches |
| 3 | Same, per token (`token_rewards`): `r_t = −β (log π_old(y_t) − log π_ref(y_t))` on every token, plus `r(x,y)` on the last token | Ouyang et al. 2022; Shao et al. 2024 | Ouyang §3.5 "Reinforcement learning (RL)": "we add a per-token KL penalty from the SFT model at each token", objective eq. (2); Shao §4.1.1 eq. (2) `r_t = r_φ(q, o_≤t) − β log(π_θ(o_t\|q,o_<t)/π_ref(o_t\|q,o_<t))` with the reward model scoring the last token | Matches. Summed over tokens it equals row 2. Ouyang eq. (2) has an extra pretraining term `γ E[log π(x)]` ("PPO-ptx"); the project is the `γ = 0` case ("PPO" in Ouyang). The penalty is evaluated at the sampling policy (`π_old`, fixed during the epochs), as in an RL reward. Stiennon et al. 2020 §3.4 states the same per-sequence reward as row 2 (unnumbered). |
| 4 | Objective `J_β(π) = E_π[r] − β KL(π ‖ π_ref)` (`exact.objective`) | Rafailov et al. 2023 | §3, eq. (3) | Matches (`r_φ` → `r`, `π_θ` → `π`) |
| 5 | Optimum `π*_β(y\|x) = π_ref(y\|x) exp(r(x,y)/β) / Z(x)` (`exact.optimum`) | Rafailov et al. 2023; Ziegler et al. 2019 | DPO §4 eq. (4), derived in App. A.1 eqs. (11)–(15); Ziegler §3.1.1: "π_opt(y\|x) ∝ ρ(y\|x) e^{r_s(x,y)/β}" | Matches. Both papers state the closed form the project uses; DPO derives it. |
| 6 | Gap identity `J_β(π*_β) − J_β(π) = β KL(π ‖ π*_β)` (`exact.gap`) | Rafailov et al. 2023 | App. A.1, eqs. (12)–(14) | Matches as a one-line consequence, not stated verbatim: eq. (14) writes `−J_β(π)/β` (per prompt) as `KL(π ‖ π*) − log Z(x)`, so `J_β(π) = β log Z(x) − β KL(π ‖ π*)`; at `π = π*` the KL is 0. A notebook should say "follows from" DPO App. A.1, not "stated in". |
| 7 | GAE: `δ_t = r_t + V_{t+1} − V_t`, `A_t = Σ_l λ^l δ_{t+l}`, `V = 0` after the last token, discount `γ = 1` (`gae`) | Schulman et al. 2016 (GAE) | §3: `δ^V_t = r_t + γV(s_{t+1}) − V(s_t)` (unnumbered, just before eq. (10)); eq. (16) `Â_t^{GAE(γ,λ)} = Σ_l (γλ)^l δ^V_{t+l}`; PPO paper eqs. (11)–(12) give the truncated form | Matches with `γ = 1`. |
| 8 | Value target `returns = A + V_old` and value loss `mean((V − returns)²)` (`gae`, `ppo_loss`) | Schulman et al. 2016; Schulman et al. 2017 | GAE §5, footnote 2: TD(λ) target `V̂^λ_t = V_φold(s_n) + Σ_l (γλ)^l δ_{t+l}` (the paper's main choice is the λ = 1 return, eq. (28)); PPO §5: `L^VF_t = (V_θ(s_t) − V^targ_t)²` | Matches (the footnote's alternative, which is what `A + V_old` is). |
| 9 | Clipped surrogate `mean(min(ρ A, clip(ρ, 1 − ε, 1 + ε) A))`, `ρ_t = π_θ(y_t)/π_old(y_t)`; total loss `−L^CLIP + c_1 L^VF` (`ppo_loss`) | Schulman et al. 2017 (PPO) | §3, eq. (7); combined objective §5, eq. (9) | Matches, with `r_t(θ)` → `ρ_t`, `c_1` = `vf_coef`, and no entropy bonus (`c_2 = 0`). |
| 10 | GRPO objective, per token: `min(ρ A, clip(ρ, 1 − ε, 1 + ε) A) − β D̂_KL`, averaged over tokens of a response then over the group (`grpo_loss`: mean over all `B·T` tokens) | Shao et al. 2024 | §4.1.1, eq. (3); App. A.1.6, eq. (19) | Matches. The paper averages `1/G Σ_i 1/\|o_i\| Σ_t`; all responses here have the same length `T`, so this equals the mean over `B·T` tokens. The KL term is in the loss, not in the reward: "instead of adding KL penalty in the reward, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss" (§4.1.1). |
| 11 | GRPO KL estimator `π_ref/π_θ − log(π_ref/π_θ) − 1`, per token (`kl_estimator`) | Shao et al. 2024 | §4.1.1, eq. (4), "the following unbiased estimator (Schulman, 2020)" | Matches exactly (claim (i) confirmed). Its gradient coefficient is given in App. A.1.6 eqs. (20)–(21): `Â_{i,t} + β(π_ref/π_θ − 1)`, which is what autograd gives on `grpo_loss`. For one next-token distribution (one state `s_t`), its expectation under samples from `π_θ` is `−β ∇KL(π_ref(·\|s_t) ‖ π_θ(·\|s_t))` (checked numerically in this task; the unit test in `tests/test_grpo.py` is per token and stands); the paper does not state this. *Amended in Task 14:* over a response sampled from `π_θ`, the expected gradient of the summed per-token terms is `Σ_t E_{s_t∼π_θ}[∇KL(π_ref(·\|s_t) ‖ π_θ(·\|s_t))]`, with `∇` acting on the conditionals only (the state distribution held fixed, since the samples are fixed). That is the gradient of neither sequence-level KL: `∇KL(π_ref ‖ π_θ)` weights states by `π_ref`. Measured by enumeration in notebook 02, §"What GRPO aims at" (a): cosines 0.73–0.92 with the reverse and 0.68–0.95 with the forward sequence-level gradient. |
| 12 | GRPO advantage `A_i = (r_i − mean(r)) / (std(r) + 1e-6)` within the group, the same for every token of `y_i` (`group_advantages`) | Shao et al. 2024 | §4.1.2: "these rewards are normalized by subtracting the group average and dividing by the group standard deviation … sets the advantages Â_{i,t} of all tokens in the output as the normalized reward, i.e., Â_{i,t} = r̃_i = (r_i − mean(r))/std(r)" | Matches (claim (ii) confirmed). The paper does not say whether `std` is the sample (n − 1) or population (n) standard deviation, nor add an `ε`; the project uses the sample std (`torch.std` default) and `ε = 1e-6`. open-instruct's GRPO does the same with `ε = 1e-8`. |
| 13 | Bradley–Terry: `p(y_1 ≻ y_2 \| x) = σ(r(x,y_1) − r(x,y_2))` | Rafailov et al. 2023 (citing Bradley and Terry 1952) | DPO §3, eq. (1) `exp(r*(x,y_1)) / (exp(r*(x,y_1)) + exp(r*(x,y_2)))`; §4 text: "p*(y_1 ≻ y_2 \| x) = σ(r*(x,y_1) − r*(x,y_2))" | Matches. The 1952 original could not be read (see "Not confirmed"). The project's preference noise, `σ((r_w − r_l)/τ)`, is the project's own; it is BT with reward `r/τ`. |
| 14 | Reward in terms of the optimum: `r(x,y) = β log(π*(y\|x)/π_ref(y\|x)) + β log Z(x)` | Rafailov et al. 2023 | §4, eq. (5); App. A.2 eq. (17) | Matches (`π_r` → `π*`). The cancellation of `Z(x)` in the BT difference is §4 eq. (6), App. A.2. |
| 15 | DPO loss `−mean(log σ(β[(log π_θ(y_w) − log π_ref(y_w)) − (log π_θ(y_l) − log π_ref(y_l))]))`, sequence log-probabilities summed over tokens (`dpo_loss`) | Rafailov et al. 2023 | §4, eq. (7) | Matches. This is the standard (summed) DPO, not the length-normalised variant of Tülu 3 eq. (6) (`β/\|y\|` on each log-ratio). |
| 16 | DPO gradient `∇L = −β E[σ(r̂(x,y_l) − r̂(x,y_w)) (∇log π_θ(y_w\|x) − ∇log π_θ(y_l\|x))]`, `r̂ = β log(π_θ/π_ref)` | Rafailov et al. 2023 | §4, "What does the DPO update do?" (unnumbered display) | Matches §4 exactly; checked against autograd on `dpo_loss` in this task. Do **not** use App. A.4's final line (see Notes). |

## Facts recorded from the GRPO paper (Shao et al. 2024)

- **β.** §4.2: "The KL coefficient is 0.04." (Also: 64 outputs per question, policy learning
  rate 1e-6.) The project's default β is a free hyperparameter of the toy world, not the paper's.
- **Is π_old refreshed every iteration?** Yes. Algorithm 1 (§4.1.1), line 6: "Update the old
  policy model π_θold ← π_θ" inside the per-step loop, followed by `μ` GRPO iterations
  (line 10) on that batch. §4.2 adds: "The policy model only has a single update following each
  exploration stage", i.e. the DeepSeekMath-RL run used `μ = 1`, so `π_θ = π_θold` when the
  gradient is taken and the clip never binds. The project's default `epochs = 4` is a
  PPO-style choice, not the paper's run.
- **Is π_ref fixed?** Not in iterative GRPO: Algorithm 1, line 3, "reference model π_ref ← π_θ"
  at the start of each outer iteration (§4.1.4). With one outer iteration (`I = 1`) it is the
  initial policy throughout, which is the project's setting.

## Facts recorded from DeepSeek-R1 and Tülu 3 (Task 14, for notebook 02's "Where it went next")

Read 2026-11-22. Quotes verbatim; a quote cut short is marked "…".

- **DeepSeek-R1 (Guo et al. 2025, Nature 645, 633–638),** read at https://www.nature.com/articles/s41586-025-09422-z.
  - Methods, "GRPO", Eqs. (1)–(3): the objective has the clipped ratio, `−β D_KL(π_θ ‖ π_ref)` in the loss with the estimator `π_ref/π_θ − log(π_ref/π_θ) − 1`, and the advantage `(r_i − mean)/std` over the group.
  - Main: "we bypass the conventional supervised fine-tuning (SFT) phase before RL training"; section "DeepSeek-R1-Zero": "we apply the RL technique on the DeepSeek-V3 Base to train DeepSeek-R1-Zero", "we use a rule-based reward system to compute accuracy and format rewards".
  - Methods, "Rule-based rewards": "…neural reward models are susceptible to reward hacking during large-scale RL".
  - Methods, "Training details of DeepSeek-R1-Zero": "we set the learning rate to 3 × 10−6, the Kullback–Leibler (KL) coefficient to 0.001"; "For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step…"; "To accelerate training, each rollout generates 8,192 outputs, which are randomly split into 16 minibatches and trained for only a single inner epoch."
  - Same section (and "Training details of the first RL stage"): "Every 400 steps, we replace the reference model with the latest policy model." So `π_ref` is refreshed during training, unlike this project's fixed `π_ref`.
- **Tülu 3 (Lambert et al. 2024),** read at https://arxiv.org/html/2411.15124v5.
  - §6: "In Tülu 3, we introduce Reinforcement Learning with Verifiable Rewards (RLVR), a novel method for training language models on tasks with verifiable outcomes such as mathematical problem-solving and instruction following."
  - §6: "We train models with RLVR following preference finetuning, and we use the PPO (Schulman et al., 2017) algorithm to optimize for the RLVR objective."
  - §6, after Eq. (8): "We set α = 10 based on pilot experiments and did not tune it further." Table 21 (§5.4.1): "The final 8B RLVR model used β=0.05 and ω=0.0". (Notebook 02 states only the PPO fact.)

## Facts recorded from Stiennon et al. 2020 and Ouyang et al. 2022 (for notebooks 01 and 04)

Read in Task 13 and again 2026-11-22, in the text extracted from the arXiv PDFs listed under
"Sources read" (Stiennon v3, Ouyang v1). Quotes verbatim; a quote cut short is marked "…".

- **Stiennon et al. 2020,** https://arxiv.org/pdf/2009.01325 (v3).
  - §3.4 "Models", paragraph "Human feedback policies": "For the PPO value function, we use a
    Transformer with completely separate parameters from the policy. This prevents updates to
    the value function from partially destroying the pretrained policy early in training (see
    ablation in Appendix G.1). We initialize the value function to the parameters of the reward
    model. In our experiments, the reward model, policy, and value function are the same size."
  - Appendix B.1 "Hyperparameters": "For PPO, we run with separate policy and value networks,
    initializing our policies to the supervised baseline, and our value functions to the reward
    model. We set γ = 1 and λ = 0.95 for the advantage estimation [57] and do 4 epochs of
    optimization for each batch of rollouts."
  - Appendix G.1 "Value function ablation": "…we conduct an ablation comparing using separate
    parameters for the value function and policy, against using a shared network … The results,
    shown in Figure 11, clearly indicate that using separate networks outperforms the latter. On
    the other hand, having separate networks increases the memory requirements of running RL
    fine-tuning."
- **Ouyang et al. 2022,** https://arxiv.org/pdf/2203.02155 (v1).
  - §3.5, paragraph "Reinforcement learning (RL)": "In addition, we add a per-token KL penalty
    from the SFT model at each token to mitigate overoptimization of the reward model. The value
    function is initialized from the RM." (The first sentence is also row 3.)
  - Appendix C.4 "Details of RLHF training": "The batch size for each iteration is 512, with a
    minibatch size of 64. In other words, each batch is randomly split into 8 minibatches and is
    trained on for only a single inner epoch (Schulman et al., 2017)." And: "As previously
    mentioned, for all PPO models we use a 6B RM and a 6B value function, and the latter is
    initialized from the former. By using the same 6B reward model and value function on
    policies of all model sizes, it’s easier to compare the effect of policy model size on
    policy performance."

What the notebooks state from these, and only that: notebook 01's carry-over note (Stiennon
B.1: γ = 1, λ = 0.95, 4 epochs; Ouyang C.4: 8 minibatches of 64 from a batch of 512, "a single
inner epoch"), its item 2 (Ouyang C.4), its "Where it went next" (Stiennon §3.4 and G.1;
Ouyang §3.5 and C.4); notebook 04's history (Stiennon §3.4: a value function with parameters
separate from the policy's, initialised from the reward model).

## Notes (paper-side points not to copy)

- **DPO App. A.4 has a sign slip.** Eq. (21) writes the loss as
  `−log σ(β log π_θ(y_l)/π_ref(y_l) − β log π_θ(y_w)/π_ref(y_w))`, the argument of eq. (7) with
  `w` and `l` swapped, and the final displayed gradient has `σ(β log π_θ(y_w)/π_ref(y_w) − β log
  π_θ(y_l)/π_ref(y_l))`, i.e. `σ(r̂_w − r̂_l)`. Autograd on eq. (7) gives `σ(r̂_l − r̂_w)`, as in
  §4. The §4 form (row 16) is right.
- **PPO eq. (11)** writes the last exponent as `(γλ)^{T−t+1}`; summing up to `δ_{T−1}` needs
  `T − t − 1`. Typo in the paper; the project's recursion is the standard one.
- **GRPO's `π_θ` in eq. (4)** is the current policy while samples come from `π_θold`; the
  "unbiased" claim holds only when the two coincide (true at `μ = 1`, the paper's run).

## Sources read

| Paper | Read from | Version |
|---|---|---|
| Williams 1992 | Author's copy, https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf (header "Appears in Machine Learning, 8, pp. 229-256, 1992"); pages read as images | preprint pagination; eq. numbers as in that copy |
| Schulman et al. 2016 (GAE) | https://arxiv.org/pdf/1506.02438 | v6 |
| Schulman et al. 2017 (PPO) | https://arxiv.org/pdf/1707.06347 | v2 |
| Ziegler et al. 2019 | https://arxiv.org/pdf/1909.08593 | v2 |
| Stiennon et al. 2020 | https://arxiv.org/pdf/2009.01325 | v3 |
| Ouyang et al. 2022 | https://arxiv.org/pdf/2203.02155 | v1 |
| Rafailov et al. 2023 (DPO) | https://arxiv.org/pdf/2305.18290 | v3 |
| Shao et al. 2024 (DeepSeekMath) | https://arxiv.org/pdf/2402.03300 | v3 |
| Lambert et al. 2024 (Tülu 3) | https://arxiv.org/pdf/2411.15124 | v5 (§5.1.2 eq. (5)–(6), Table 20; §6 and Table 21 read in Task 14, HTML version) |
| Guo et al. 2025 (DeepSeek-R1) | https://www.nature.com/articles/s41586-025-09422-z | Nature 645, 633–638 (Methods, read in Task 14) |

## Not confirmed

- **Bradley and Terry 1952**, the original text: JSTOR and OUP pages could not be fetched.
  The BT form is confirmed only as DPO eq. (1) states it; bibliographic fields come from the
  Crossref records for DOIs 10.2307/2334029 and 10.1093/biomet/39.3-4.324.
- **Williams 1992**, the journal version: equation numbers were read from the author's
  preprint copy, which states it appears in Machine Learning 8, pp. 229–256. The journal's own
  equation numbering was not seen.
