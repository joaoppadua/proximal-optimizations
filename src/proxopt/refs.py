"""The papers the notebooks cite, each field copied from the page in `url` (or, where `url` is a
DOI, from that DOI's Crossref record). Notebooks cite only through `cite`.

Authors: up to six as listed on the source page, then "et al.". `venue` is the publication
venue of the cited version when its source names one, otherwise "arXiv preprint
arXiv:NNNN.NNNNN". `year` is the venue's year when a venue is named, otherwise the year of
first submission. (olmo2024 cites the full arXiv version, a preprint first submitted in 2024;
its shorter COLM 2025 version is mentioned in `venue` but is not the version cited.)"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ref:
    authors: str
    year: int
    title: str
    venue: str
    url: str


REFS: dict[str, Ref] = {
    "williams1992": Ref(
        authors="Ronald J. Williams",
        year=1992,
        title="Simple statistical gradient-following algorithms for connectionist "
        "reinforcement learning",
        venue="Machine Learning 8(3-4), 229-256",
        url="https://doi.org/10.1007/BF00992696",
    ),
    "schulman2015trpo": Ref(
        authors="John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel",
        year=2015,
        title="Trust Region Policy Optimization",
        venue="ICML 2015",
        url="https://arxiv.org/abs/1502.05477",
    ),
    "schulman2016gae": Ref(
        authors="John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel",
        year=2016,
        title="High-Dimensional Continuous Control Using Generalized Advantage Estimation",
        venue="ICLR 2016",
        url="https://arxiv.org/abs/1506.02438",
    ),
    "schulman2017ppo": Ref(
        authors="John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov",
        year=2017,
        title="Proximal Policy Optimization Algorithms",
        venue="arXiv preprint arXiv:1707.06347",
        url="https://arxiv.org/abs/1707.06347",
    ),
    "christiano2017": Ref(
        authors="Paul Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, "
        "Dario Amodei",
        year=2017,
        title="Deep reinforcement learning from human preferences",
        venue="Advances in Neural Information Processing Systems (NIPS 2017)",
        url="https://arxiv.org/abs/1706.03741",
    ),
    "ziegler2019": Ref(
        authors="Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, "
        "Dario Amodei, et al.",
        year=2019,
        title="Fine-Tuning Language Models from Human Preferences",
        venue="arXiv preprint arXiv:1909.08593",
        url="https://arxiv.org/abs/1909.08593",
    ),
    "stiennon2020": Ref(
        authors="Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, "
        "Chelsea Voss, et al.",
        year=2020,
        title="Learning to summarize from human feedback",
        venue="NeurIPS 2020",
        url="https://arxiv.org/abs/2009.01325",
    ),
    "ouyang2022": Ref(
        authors="Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, "
        "Pamela Mishkin, et al.",
        year=2022,
        title="Training language models to follow instructions with human feedback",
        venue="Advances in Neural Information Processing Systems 35 (NeurIPS 2022)",
        url="https://arxiv.org/abs/2203.02155",
    ),
    "bradley1952": Ref(
        authors="Ralph Allan Bradley, Milton E. Terry",
        year=1952,
        title="Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons",
        venue="Biometrika 39(3/4), 324-345",
        url="https://doi.org/10.2307/2334029",
    ),
    "rafailov2023": Ref(
        authors="Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, "
        "Christopher D. Manning, Chelsea Finn",
        year=2023,
        title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
        venue="Advances in Neural Information Processing Systems 36 (NeurIPS 2023)",
        url="https://arxiv.org/abs/2305.18290",
    ),
    "shao2024": Ref(
        authors="Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, et al.",
        year=2024,
        title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language "
        "Models",
        venue="arXiv preprint arXiv:2402.03300",
        url="https://arxiv.org/abs/2402.03300",
    ),
    "guo2025": Ref(
        authors="Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, "
        "et al.",
        year=2025,
        title="DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
        venue="Nature 645(8081), 633-638",
        url="https://doi.org/10.1038/s41586-025-09422-z",
    ),
    "lambert2024": Ref(
        authors="Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, "
        "Hamish Ivison, Faeze Brahman, et al.",
        year=2024,
        title="Tulu 3: Pushing Frontiers in Open Language Model Post-Training",
        venue="arXiv preprint arXiv:2411.15124",
        url="https://arxiv.org/abs/2411.15124",
    ),
    "olmo2024": Ref(
        authors="Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, "
        "et al.",
        year=2024,
        title="2 OLMo 2 Furious",
        venue="arXiv preprint arXiv:2501.00656 (a shorter version accepted to COLM 2025)",
        url="https://arxiv.org/abs/2501.00656",
    ),
}


def _format(ref: Ref) -> str:
    return f"{ref.authors} ({ref.year}). {ref.title}. {ref.venue}. <{ref.url}>"


def cite(*keys: str) -> str:
    """A markdown bulleted list, one formatted reference per key, in the order given.
    An unknown key raises KeyError."""
    return "\n".join(f"- {_format(REFS[k])}" for k in keys)
