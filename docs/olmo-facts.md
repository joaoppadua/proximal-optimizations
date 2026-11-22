# OLMo 2 1B (April 2025): two facts checked at the source

Checked 2026-11-21 (Task 10). Each answer quotes its source verbatim, with the URL read.

## 1. Did the last post-training stage of OLMo 2 1B use GRPO?

**Yes.**

The OLMo 2 paper, §5 "Deep Dive: Post-training Pipeline", paragraph "Reinforcement Learning with
Verifiable Rewards (RLVR)",
read from https://arxiv.org/pdf/2501.00656 (v3, page 28):

> For the 1B and 32B model, we performed RLVR with Group Relative Policy Optimization (GRPO)
> (Shao et al., 2024), which forgoes the need for a reward model.

The 7B and 13B models used PPO instead (same page: "For RLVR, we initialize PPO's value
function from the corresponding RMs").

The open-instruct repository documents the 1B runs, which show that the RLVR stage was two GRPO
runs in sequence and that the last of them produced the Instruct checkpoint.
https://github.com/allenai/open-instruct/blob/a4cd966a9e202843baba653115045e6fd6f473db/docs/olmo2.md
("RLVR" → "1B"):

> The 1B OLMo 2 model has two RL stages run in sequence. The first is on MATH, GSM8K, and IF
> constraints.

Then "Next, on MATH only", the original command for the second stage is
`python open_instruct/grpo_vllm_thread_ray_gtrl.py --exp_name 0427_grpo_seed_1_lr_9e-7 --beta 0.01 ... --kl_estimator kl3 ... --dataset_mixer_list allenai/RLVR-MATH 1.0 ... --model_name_or_path allenai/OLMo-2-0425-1B-RLVR1`.
The model cards agree on the lineage: https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct
lists "Finetuned from model: allenai/OLMo-2-0425-1B-RLVR1" and "final RLVR training on [this
dataset](https://huggingface.co/datasets/allenai/RLVR-MATH)"; the base card
https://huggingface.co/allenai/OLMo-2-0425-1B lists post-training as "SFT+DPO+GRPO".

Details of that GRPO, from the same sources (for notebook 04, if useful):
- β = 0.01 in both 1B stages (`--beta 0.01`, docs/olmo2.md above), not DeepSeekMath's 0.04.
- KL estimator `kl3`. In the script those runs used,
  https://github.com/allenai/open-instruct/blob/745bf58d321c/open_instruct/grpo_vllm_thread_ray_gtrl.py,
  `kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff` with
  `ref_logprobs_diff = new_logprobs - ref_logprobs`, which is `π_ref/π_θ − log(π_ref/π_θ) − 1`,
  the estimator of Shao et al. eq. (4), and it is added to the loss ("grpo change: directly
  subtract KL in loss").
- Advantages in that script: `(scores - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)`
  over groups of `number_samples_per_prompt = 16`.

## 2. Did its DPO stage use the length-normalised variant, and with which β?

**Length-normalised, β = 5.**

https://github.com/allenai/open-instruct/blob/a4cd966a9e202843baba653115045e6fd6f473db/docs/olmo2.md
("Preference Tuning (DPO)" → "1B"), the command for run `0424_1B_dpo_onpol_lr_2.5e-6_seed_111`
on `--model_name_or_path allenai/OLMo-2-0425-1B-SFT` with
`--dataset_mixer_list allenai/olmo-2-0425-1b-preference-mix 1.0`, ends with:

> `--dpo_loss_type dpo_norm \`
> `--dpo_beta 5`

What `dpo_norm` computes, read in the code at commit 745bf58d321c, the commit the same
`docs/olmo2.md` names for the 1B RLVR experiments of the same weeks (run names dated 0423 and
0427; the DPO run is dated 0424). The DPO run's own commit is not documented. In
https://github.com/allenai/open-instruct/blob/745bf58d321c/open_instruct/dpo_tune_cache.py,
lines 793-794:

> `average_log_prob_loss_types = ["simpo", "dpo_norm"]`
> `average_log_prob = args.dpo_loss_type in average_log_prob_loss_types`

and in https://github.com/allenai/open-instruct/blob/745bf58d321c/open_instruct/dpo_utils.py,
`_get_batch_logps`, lines 170-171:

> `if average_log_prob:`
> `    return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)`

that is, the per-token mean log-probability of each response, which then enters the usual
`dpo_loss` (line 73, quoted up to the label-smoothing term, which is 0 by default:
`losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing)`).
This is Tülu 3 eq. (6)
(https://arxiv.org/pdf/2411.15124, v5, §5.1.2):

> We find (in Section 5.4) that length-normalized DPO works best

Caveats: this β = 5 multiplies per-token mean log-ratios, so it is not comparable with a β on
summed log-ratios. The source is the repository's documentation of the command, published by the
model's authors; the OLMo 2 paper itself does not name the DPO variant or β, and neither do the
model cards. The 1B DPO data is the one named on the DPO model card
(https://huggingface.co/allenai/OLMo-2-0425-1B-DPO: "further DPO training on [this
dataset](https://huggingface.co/datasets/allenai/olmo-2-0425-1b-preference-mix)"), matching the
command.

GRPO_CONFIRMED: yes
DPO_VARIANT: length-normalised
