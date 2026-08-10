# NOTICE — MRBench dataset (vendored)

`MRBench_V1.json` in this directory is a copy of the **MRBench** benchmark
(version V1) released with the paper cited below. It is redistributed here under
its original license, **CC BY-SA 4.0**.

## Source

- **Dataset:** MRBench (`MRBench_V1.json`) — 192 conversations / 1,596 responses,
  8 pedagogical dimensions.
- **Official repository:**
  https://github.com/kaushal0494/UnifyingAITutorEvaluation
  (file `MRBench/MRBench_V1.json`).
- **Paper:** Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova, and
  Ekaterina Kochmar. *"Unifying AI Tutor Evaluation: An Evaluation Taxonomy for
  Pedagogical Ability Assessment of LLM-Powered AI Tutors."* NAACL 2025 (Long
  Papers), pp. 1234–1251. https://aclanthology.org/2025.naacl-long.57/ ,
  DOI `10.18653/v1/2025.naacl-long.57`.

## License — CC BY-SA 4.0

MRBench is released under the **Creative Commons Attribution-ShareAlike 4.0
International (CC BY-SA 4.0)** license:
https://creativecommons.org/licenses/by-sa/4.0/

This license requires:

1. **Attribution** — credit the creators (see suggested text and citation below),
   provide a link to the license, and indicate if changes were made.
2. **ShareAlike** — if you remix, transform, or build upon the material, you must
   distribute your contributions **under the same CC BY-SA 4.0 license**.

**Any redistributed derivative of this data** (filtered subsets, reformatted
copies, or artifacts that embed MRBench content) **must remain under CC BY-SA
4.0** and carry this attribution.

> Source of the license fact: the "License" section of the official repository
> README (verified 2026-08-09). The NAACL paper PDF in `_paper_refs/` does not
> itself restate the dataset license, so the authoritative statement is the repo.

## Upstream datasets

MRBench builds upon two prior datasets; credit them alongside MRBench and note
that their terms flow through the share-alike chain:

- **MathDial** — https://github.com/eth-nlped/mathdial
- **Bridge** — https://github.com/rosewang2008/bridge

## Required attribution (suggested text)

> MRBench dataset © Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova,
> and Ekaterina Kochmar (MBZUAI), from "Unifying AI Tutor Evaluation" (NAACL
> 2025), used under CC BY-SA 4.0. Built upon the MathDial and Bridge datasets.

## Citation

```bibtex
@inproceedings{maurya-etal-2025-unifying,
  title = "Unifying {AI} Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of {LLM}-Powered {AI} Tutors",
  author = "Maurya, Kaushal Kumar and Srivatsa, Kv Aditya and Petukhova, Kseniia and Kochmar, Ekaterina",
  booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
  month = apr,
  year = "2025",
  address = "Albuquerque, New Mexico",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.naacl-long.57/",
  doi = "10.18653/v1/2025.naacl-long.57",
  pages = "1234--1251",
  ISBN = "979-8-89176-189-6"
}
```

## Local provenance

- `MRBench_V1.json` — the V1 benchmark file, used as-is. See
  `Plan/mrbench/README.md` for the accepted data-vs-paper (Table 3) divergence.
- `_paper_refs/` — local copies of the paper and annotation-guideline PDFs kept
  for reference only. They are **git-ignored** and are **not** redistributed.
