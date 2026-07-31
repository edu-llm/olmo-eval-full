Abstract
According to estimates by AI labs, benchmarking and evaluating takes around 10% to 15% of total model training time (Olmo-3, Llama). Fast, granular, and trustworthy benchmarking is crucial for EDU-LLM due to the sheer number of novel hypotheses being tested. We propose a skill-based Computerized Adaptive Test, showing a reduction in the number of items required in a diagnostic to report a predicted accuracy on a specific skill in benchmarks, while holding a similar level of precision as full-length benchmarks. Our findings are split between Multiple Choice benchmarks and Free Response benchmarks. 
MCQ
On MCQ benchmarks, we reproduce ATLAS style adaptive testing on__. Using __ calibration models in the 0-7B range(?), CAT recovers full benchmark accuracy with __% of items
FRQ
On FRQ benchmarks, we grade responses with an LLM-as-judge against per scenario rubrics to build a criterion level response matrix, derive a Q-matrix, and fit 2PL/MIRT so CAT selects scenarios adaptively.
MCQ
Strategy
Our work builds upon the research from ATLAS, which showed the potential for Adaptive Testing to create dramatic reduction in the amount of required items for predicting benchmark accuracy. In addition, items calibrated using Item Response Theory have been shown to provide additional information beyond accuracy. Models that get harder questions right, and miss certain easy questions are appropriately ranked higher when using CAT, despite a simple accuracy comparison being unable to highlight this difference. 
Our paper showcases the feasibility of recreating ATLAS methodology on new benchmarks, and specifically formalizes the requirements to use CAT as a true MCQ benchmark replacement.  This paper highlights the promises and shortcomings of MIRT for MCQ, a discussion largely missing from ATLAS. 
ATLAS Recreation
Intro
While ATLAS showcases promising results for reduction in benchmark size, there is a gap between ATLAS results and actual applications for evals during model training. The ability to apply IRT calibration to a benchmark without public response data, requires a proper analysis of the upfront cost required to get a desired correlation. Also, the ability for the benchmarks to discriminate between models at parameter ranges without a lot of calibration models was not explicitly shown.
Feasibility
Two main experiments were performed for feasibility. Calibration on Pedagogy Benchmark, a very specific skill based benchmark with lack of publicly available response data. Calibration using subsets of ATLAS data to mimic Real-World scenarios and showcase the trade-offs between model diversity, count, and parameter-range match on calibration correlation. 
The inference cost is given at the 0 - 7bn param range ->

L4 GPUS were used for inference, and CPUS were used to parallelize model downloads to accessible cache.
There are two points of discussion, 1 the reduction of time during inference, allowing for use of CAT for checkpointing, and faster experimentation. Also, the actual delta in cost.
Availability and Quality of Response Data
OpenLM, OpenHelm
2 of the more higher quality sources of model responses per question on benchmarks. OpenLM also used by ATLAS
The variability in prompting strategy/other issues
Parameter Distribution
Benchmarks take time to get added to open LM
Experiments
Grounding the Hypothesis in Theory:
IRT calibration discussion
Importance of Model Diversity, Reducing Model Count While retaining Correlation
Importance of Parameter Range
Reducing Models by specifying Range
Using out of range models
Results
MIRT Discussion
Conclusion
FRQ
Strategy - See Report
LLM-as-judge
Full Selection Process
Prometheus 2 had previously been used without first proving that its judgments matched human graders.
We graded three tutor-model responses for each of 10 scenarios. Each response was graded separately against its corresponding criteria, producing a total of 261 binary response-criterion cases
Success Thresholds we aimed for:
Macro-F1 >= 0.80 - The judge must perform well on both Pass and Fail cases, without good performance on one class hiding poor performance on the other
Critical Failure Sensitivity >= 0.9 - The judge must detect at least 90% of serious failures identified by humans
Every mapped skills F1 >= 0.70 - The judge must perform reasonably well in every tutoring-skill category, not only overall
Repeat agreement >= 0.90 - When given the same response multiple times, the judge must return the same verdict at least 90% of the time
Prompt flip rate <= 0.10 - Tiny variances to the prompt passed to the judge don’t matter. Small, meaning-preserving prompt changes must alter theudge’s verdict no more than 10% of cases
Blind the judge evaluation: Judges received the scenario, tutor response, and criterion, but not the human label or tutor identity (for anonymity to reduce bias)
Use each models’ appropriate format
Prometheus retained its 1-5 scoring scoring system
Flow and Selene used their native binary formats
Qwen and Gemma were instructed to return binary JSON judgments
Each local judge ran in six waves: three identical runs to measure test-retest consistency, and then three harmless prompt variations: whitespace, renamed headers, and more polite instructions
The comparison code made no additional LLM calls. It joined the stored judge results to the human labels and calculated accuracy, Maco-F1, critical sensitivity, per-skill performance, coverage, repeat agreement, and prompt flips
Parsing failures were separated from genuine grading errors. Malformed outputs were recovered, while ambiguous outputs became “ision”. Qwen’s disagreements were manually reviewed: some were real judge errors, while others revealed questionable human labels or unclear grading policies
The strongest local models (Qwen, Flow, and Selene) were rerun using stricter evidence requirements. Zero-shot, few-shot, and countercheck prompting were tested on a 96-case pilot
Zero-shot: The judge receives the rules, criterion, and tutor response but no example judgments
Few-shot: the same prompt also includes several examples showing how to handle partial answers, rounding, reference leakage, hint-only requirements, etc.
Countercheck: Includes the few-shot examples, then instructs the judge to actively look for a missing clause, wrong value, contradiction, or other reason to overturn a tentative pass. It is supposed to be stricter.
Select and freeze Qwen: final configuration was Qwen/Qwen3.5-9B zero-shot binary judging, one criterion at a time, evidence required from tutor responses, and a p_fail >= 0.33 meant fail
Results

No judge passed every threshold. Qwen was the best practical local candidate
Each frontier judge graded only 174 cases because it was prevented from grading tutor responses from its own provider family
Repeat and prompt-stability tests were performed during earlier screening. The final zero-shot, p_fail = 0.33 configuration completed only one canonical development run, not the the complete six-wave reliability test
Why Qwen instead of a frontier model?
Although the frontier judges produced similar results, none passed every acceptance threshold. Qwen was selected because it had the strongest overall performance among the local candidates, was highly stable during candidate screening, and could be self-hosted with a frozen checkpoint. This provides greater reproducibility, privacy, deployment control, and lower marginal cost for the hundreds of thousands of judgments required during calibration.
skills/rubrics
MIRT needs a set of skills for each benchmark and a Q-matrix that records the skills each criterion tests. We calibrate several ability scores where the benchmark supports it. To build the Q-matrix, we write a definition for each skill and label the criteria against those definitions, then review some of the labels by hand.
TutorBench (662 scenarios and 6,462 criteria across content, diagnosis, and scaffolding), TutorEval (828 scenarios; conceptual and quantitative), WildBench (1,001 general free response scenarios with eleven capability tags), EduBench (nine task types graded by deterministic verifiers), and Bridge (642 scenarios, with a rubric from 39 templates and five tutoring skills).
Feasibility
For calibration, first 82 open models under 7B parameters answer all 662 TutorBench scenarios to form the response matrix, generated on a single L4 GPU. Then the frozen judge grades each answer criterion by criterion.
A second run of 52 models were ran on __, with 19 overlapping between the original 82 and the new 52. 
For reproduction, the smallest models often fall into repetitive loops that the judge marks unscorable, while long inputs are also capped at 32k, which reaches close to complete coverage.
Inference Cost For Tutor Models
Parameter Range
(Billions of Parameters)
Average Individual Inference Time on L40S GPU (Seconds)
0 - 1
4
1 - 2
10
2 - 5
19
5 - 7
34



MIRT results
TutorBench:
Calibration Conditions:
115 Models ran, 2 Skill Model(correctness + scaffolding), Batch EAP / MWLE ability estimates. Presentation reported separately
H
H
H
H
Limitations
Standard Error of parameters on 115 models gets added in to any skill estimate standard error, with more models for calibration we could reduce it down from around .2 average to something more reasonable. 
Judge isn’t perfect and doesn’t necessarily agree with human graders in many cases



