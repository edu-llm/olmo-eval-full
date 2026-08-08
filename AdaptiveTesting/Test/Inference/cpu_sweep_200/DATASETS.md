# Dataset download links (CPU sweep suite)

All loaders use Hugging Face `datasets`. Prefetch with `prefetch_datasets.py`.

| Suite name | Role | Hugging Face | Notes |
|---|---|---|---|
| **OpenBookQA** | MCQ | https://huggingface.co/datasets/allenai/openbookqa | config `main`, split `test` |
| **SocialQA** (Social IQa) | MCQ | https://huggingface.co/datasets/allenai/social_i_qa | parquet revision; split `validation` |
| **PIQA** | MCQ | https://huggingface.co/datasets/ybisk/piqa | `revision=refs/convert/parquet`, split `validation` |
| **PedagogyBench** | MCQ | https://huggingface.co/datasets/AI-for-Education/pedagogy-benchmark | config `cdpk_main`, split `train` (may be gated) |
| **Bridge** | open | https://huggingface.co/datasets/rose-e-wang/bridge | split `validation` |
| **EduBench** | open | https://huggingface.co/datasets/DirectionAI/EduBench | split `test` |
| **BiGGen** | open | https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench | split `test` |
| **IFEval** | open | https://huggingface.co/datasets/google/IFEval | split `train` (~541 prompts) |
| **InFoBench** | open | https://huggingface.co/datasets/kqsong/InFoBench | split `train` |
| **TutorBench** | open | https://huggingface.co/datasets/tutorbench/tutorbench | split `train` |
| **TutorEval** | open | https://huggingface.co/datasets/princeton-nlp/TutorEval | split `train` |
| **WildBench** | open | https://huggingface.co/datasets/allenai/WildBench | config `v2`, split `test` |

MCQ outputs record `result=correct|wrong`. Open outputs record raw `response` only (`--no-judge`).

Registry aliases: `openbookqa`, `socialiqa`, `piqa`, `pedagogy`, `bridge`, `edubench`, `biggen`, `ifeval`, `infobench`, `tutorbench`, `tutoreval`, `wildbench` — or `--benchmarks cpu_sweep`.
