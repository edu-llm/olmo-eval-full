# Pilot curation change report

- policy: `curation_v1`
- source rubrics: `data/rubrics_qmatrix_final.jsonl` (6462 criteria)
- curated rubrics: `data\curated\rubrics_qmatrix_curated.jsonl` (6845 criteria)
- scenarios edited by ops: 63
- scenarios changed (incl. presentation consolidation): 342
- edit operations logged: 1351

## Ops summary
- format_consolidated: 313
- keep: 536
- presentation_added: 349
- rescope_optional: 3
- soften: 8
- split: 142

## Per-change detail

### tb_0001
- **rescope_optional** `tb_0001_c02` -> `tb_0001_c02`
    - was: The response must include the formula to be used to calculate the ionic concentration. (e.g., pH to [H⁺] conversion formula: $[H^+] = 10^{-pH}$)
    - now: If the response re-derives the ionization from scratch, it must include the pH→[H⁺] conversion formula ([H⁺] = 10^(−pH)).
- **keep** `tb_0001_c04` -> `tb_0001_c04`
- **rescope_optional** `tb_0001_c07` -> `tb_0001_c07`
    - was: The response must include the formula to be used to calculate percentage ionization. (e.g., % ionization = [H⁺] or [A⁻] / [HA] initial × 100)
    - now: If the response re-derives the percent ionization, it must include the percent-ionization formula (e.g., % ionization = [H⁺]/[HA]ᵢₙᵢₜᵢₐₗ × 100).
- **rescope_optional** `tb_0001_c08` -> `tb_0001_c08`
    - was: The response must include the formula used to calculate Ka. (e.g., K_a = \frac{[H^+][A^-]}{[HA]} \]))
    - now: If the response re-derives Ka, it must include the Ka expression (Ka = [H⁺][A⁻]/[HA]).

### tb_0002
- **keep** `tb_0002_c03` -> `tb_0002_c03`

### tb_0003
- **split** `tb_0003_c09` -> `tb_0003_c09`
    - was: The response must provide explanations to answer the student's question, i.e. you made 3 errors in you solution, \(\frac{du}{dx} = \frac{dy}{dx}\), \[
\int \frac{du}{\cos(u)} = \si...
    - now: The response must identify and explain the first error in the student's solution: \(\frac{du}{dx} = \frac{dy}{dx}\) is incorrect.
- **split** `tb_0003_c09` -> `tb_0003_c10`
    - was: The response must provide explanations to answer the student's question, i.e. you made 3 errors in you solution, \(\frac{du}{dx} = \frac{dy}{dx}\), \[
\int \frac{du}{\cos(u)} = \si...
    - now: The response must identify and explain the second error: \(\int \frac{du}{\cos(u)} = \sin(u)\) is incorrect.
- **split** `tb_0003_c09` -> `tb_0003_c11`
    - was: The response must provide explanations to answer the student's question, i.e. you made 3 errors in you solution, \(\frac{du}{dx} = \frac{dy}{dx}\), \[
\int \frac{du}{\cos(u)} = \si...
    - now: The response must identify and explain the third error: \(\int dx = Cx\) is incorrect (which, together with the other errors, leads to the incorrect final answer \(y = \arcsin(Cx) - x\)).
- **format_consolidated** `tb_0003_c10` -> `tb_0003_c12`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0004
- **split** `tb_0004_c02` -> `tb_0004_c02`
    - was: The response must convert units for all PV = nRT variables (P_O_2 to atom, 950 mL -> 0.950 L, 25°C -> 298 K) and calculate n_O_2 ≈ 3.74 x 10^-2 mol using R = 0.08206 L atm mol^-1 K...
    - now: The response must convert the given quantities to PV=nRT units (pressure to atm, 950 mL → 0.950 L, 25°C → 298 K).
- **split** `tb_0004_c02` -> `tb_0004_c03`
    - was: The response must convert units for all PV = nRT variables (P_O_2 to atom, 950 mL -> 0.950 L, 25°C -> 298 K) and calculate n_O_2 ≈ 3.74 x 10^-2 mol using R = 0.08206 L atm mol^-1 K...
    - now: The response must compute n(O₂) ≈ 3.74×10⁻² mol using R = 0.08206 L·atm·mol⁻¹·K⁻¹.
- **format_consolidated** `tb_0004_c08` -> `tb_0004_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0005
- **split** `tb_0005_c03` -> `tb_0005_c03`
    - was: The response must prove the formulas are equivalent by first explicitly stating the definition tan(θ) = sin(θ)/cos(θ) and then showing the step-by-step simplification.
    - now: The response must state the definition tan(θ) = sin(θ)/cos(θ).
- **split** `tb_0005_c03` -> `tb_0005_c04`
    - was: The response must prove the formulas are equivalent by first explicitly stating the definition tan(θ) = sin(θ)/cos(θ) and then showing the step-by-step simplification.
    - now: The response must show the step-by-step simplification proving the two formulas are equivalent.

### tb_0006
- **keep** `tb_0006_c02` -> `tb_0006_c02`
- **keep** `tb_0006_c03` -> `tb_0006_c03`

### tb_0007
- **keep** `tb_0007_c05` -> `tb_0007_c05`

### tb_0008
- **keep** `tb_0008_c04` -> `tb_0008_c04`
- **format_consolidated** `tb_0008_c06,tb_0008_c07` -> `tb_0008_c06`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0009
- **keep** `tb_0009_c01` -> `tb_0009_c01`

### tb_0010
- **split** `tb_0010_c01` -> `tb_0010_c01`
    - was: The response must identify and explain the addition rule for mutually exclusive events as the main missing background knowledge, stating that P(A or B) = P(A) + P(B) when events ca...
    - now: The response must identify the addition rule for mutually exclusive events as the key missing background knowledge.
- **split** `tb_0010_c01` -> `tb_0010_c02`
    - was: The response must identify and explain the addition rule for mutually exclusive events as the main missing background knowledge, stating that P(A or B) = P(A) + P(B) when events ca...
    - now: The response must state the rule: P(A or B) = P(A) + P(B) when the events cannot occur simultaneously.
- **format_consolidated** `tb_0010_c07` -> `tb_0010_c08`
    - was: 1 orphan(s) -> aspects=['generic']
    - now: The response should follow tutoring presentation conventions: be clearly and readably presented. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0011
- **split** `tb_0011_c01` -> `tb_0011_c01`
    - was: The response must explicitly identify and explain the core conceptual distinction by using the terms "experimental unit" and "matching unit" (or "blocking unit"), clarifying that t...
    - now: The response must identify the core conceptual distinction between an 'experimental unit' and a 'matching/blocking unit'.
- **split** `tb_0011_c01` -> `tb_0011_c02`
    - was: The response must explicitly identify and explain the core conceptual distinction by using the terms "experimental unit" and "matching unit" (or "blocking unit"), clarifying that t...
    - now: The response must explain that this distinction is the root of the student's confusion.
- **split** `tb_0011_c03` -> `tb_0011_c04`
    - was: The response must explicitly define "experimental unit" as the smallest entity to which treatment is independently applied, and then clearly explain why each of the 72 individual p...
    - now: The response must define 'experimental unit' as the smallest entity to which a treatment is independently applied.
- **split** `tb_0011_c03` -> `tb_0011_c05`
    - was: The response must explicitly define "experimental unit" as the smallest entity to which treatment is independently applied, and then clearly explain why each of the 72 individual p...
    - now: The response must explain why each of the 72 individuals is an experimental unit (treatment applied independently to each).
- **soften** `tb_0011_c08` -> `tb_0011_c10`
    - was: The response must present the explanation in a pedagogically optimal logical order that gradually builds understanding for a confused student, explicitly following a structure simi...
    - now: The response must present the explanation in a pedagogically effective logical order that gradually builds understanding for a confused student (e.g., clarify the distinction first, then define/explain the roles, then refute the misconception with an example, then summarize).
- **format_consolidated** `tb_0011_c10` -> `tb_0011_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0012
- **format_consolidated** `tb_0012_c07` -> `tb_0012_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0013
- **keep** `tb_0013_c02` -> `tb_0013_c02`
- **split** `tb_0013_c03` -> `tb_0013_c03`
    - was: The response must state the multiplication rule for conditional independence: P(A and B | C) = P(A | C) x P(B | C) and provide the actual calculations showing 0.92 x 0.88 = 0.8096.
    - now: The response must state the multiplication rule for conditional independence: P(A and B | C) = P(A | C) × P(B | C).
- **split** `tb_0013_c03` -> `tb_0013_c04`
    - was: The response must state the multiplication rule for conditional independence: P(A and B | C) = P(A | C) x P(B | C) and provide the actual calculations showing 0.92 x 0.88 = 0.8096.
    - now: The response must provide the calculation showing 0.92 × 0.88 = 0.8096.

### tb_0014
- **split** `tb_0014_c02` -> `tb_0014_c02`
    - was: The response must describe the two-step activation of Kinase Z: (a) allosteric release of the autoinhibitory tail when Z binds Protein Y, then (b) phosphorylation of Thr 183 in Z's...
    - now: The response must describe step (a): allosteric release of the autoinhibitory tail when Kinase Z binds Protein Y.
- **split** `tb_0014_c02` -> `tb_0014_c03`
    - was: The response must describe the two-step activation of Kinase Z: (a) allosteric release of the autoinhibitory tail when Z binds Protein Y, then (b) phosphorylation of Thr 183 in Z's...
    - now: The response must describe step (b): phosphorylation of Thr 183 in Z's activation loop by the helper kinase CK-α.
- **format_consolidated** `tb_0014_c08` -> `tb_0014_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0015
- **keep** `tb_0015_c02` -> `tb_0015_c02`
- **split** `tb_0015_c07` -> `tb_0015_c07`
    - was: The response must acknowledge that the original value for the torque is incorrect, because the torque from the friction should be subtracted, not added (i.e., $\tau_{net}=\tau_1-\t...
    - now: The response must acknowledge that the student's original value for the net torque is incorrect.
- **split** `tb_0015_c07` -> `tb_0015_c08`
    - was: The response must acknowledge that the original value for the torque is incorrect, because the torque from the friction should be subtracted, not added (i.e., $\tau_{net}=\tau_1-\t...
    - now: The response must explain the correction: the friction torque should be subtracted, not added, giving \(\tau_{net} = \tau_1 - \tau_2\).
- **format_consolidated** `tb_0015_c09` -> `tb_0015_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0017
- **keep** `tb_0017_c05` -> `tb_0017_c05`
- **keep** `tb_0017_c08` -> `tb_0017_c08`
- **keep** `tb_0017_c18` -> `tb_0017_c18`

### tb_0018
- **format_consolidated** `tb_0018_c04` -> `tb_0018_c06`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0019
- **split** `tb_0019_c01` -> `tb_0019_c01`
    - was: The response must correctly identify that \( X \sim \text{Binomial}(n = 10, p = 0.35) \) is a valid approximation, and explicitly justify this using the condition \( \frac{n}{N} < ...
    - now: The response must identify that X ~ Binomial(n = 10, p = 0.35) is a valid approximation.
- **split** `tb_0019_c01` -> `tb_0019_c02`
    - was: The response must correctly identify that \( X \sim \text{Binomial}(n = 10, p = 0.35) \) is a valid approximation, and explicitly justify this using the condition \( \frac{n}{N} < ...
    - now: The response must justify this using the condition n/N < 0.10 (with N = 120).
- **split** `tb_0019_c01` -> `tb_0019_c03`
    - was: The response must correctly identify that \( X \sim \text{Binomial}(n = 10, p = 0.35) \) is a valid approximation, and explicitly justify this using the condition \( \frac{n}{N} < ...
    - now: The response must note that the exact distribution is hypergeometric (sampling without replacement) and explain why the binomial approximation is acceptable.
- **soften** `tb_0019_c02` -> `tb_0019_c04`
    - was: The response must declare the random variable as \( X \sim \text{Binomial}(n = 10, p = 0.35) \) using formal notation and define both \( n = 10 \) (sample size) and \( p = 0.35 \) ...
    - now: The response must declare the random variable as \( X \sim \text{Binomial}(n = 10, p = 0.35) \) using formal notation and define both \( n = 10 \) (sample size) and \( p = 0.35 \) (proportion working over 40 hours) in context.
- **split** `tb_0019_c03` -> `tb_0019_c05`
    - was: The response must explicitly show that \( P(X \geq 2) = 1 - P(X = 0) - P(X = 1) \) and must calculate both \( P(X = 0) \) and \( P(X = 1) \) separately. Failure to show this comple...
    - now: The response must use the complement identity P(X≥2) = 1 − P(X=0) − P(X=1).
- **split** `tb_0019_c03` -> `tb_0019_c06`
    - was: The response must explicitly show that \( P(X \geq 2) = 1 - P(X = 0) - P(X = 1) \) and must calculate both \( P(X = 0) \) and \( P(X = 1) \) separately. Failure to show this comple...
    - now: The response must compute P(X=0).
- **split** `tb_0019_c03` -> `tb_0019_c07`
    - was: The response must explicitly show that \( P(X \geq 2) = 1 - P(X = 0) - P(X = 1) \) and must calculate both \( P(X = 0) \) and \( P(X = 1) \) separately. Failure to show this comple...
    - now: The response must compute P(X=1).
- **split** `tb_0019_c04` -> `tb_0019_c08`
    - was: The response must correctly apply the binomial formula:  
\[
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
\]  
and must do so for both \( k = 0 \) and \( k = 1 \), showing all terms...
    - now: The response must apply the binomial pmf P(X=k) = C(n,k) p^k (1−p)^(n−k) for k = 0, showing the terms.
- **split** `tb_0019_c04` -> `tb_0019_c09`
    - was: The response must correctly apply the binomial formula:  
\[
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
\]  
and must do so for both \( k = 0 \) and \( k = 1 \), showing all terms...
    - now: The response must apply the binomial pmf for k = 1, showing the terms.
- **soften** `tb_0019_c05` -> `tb_0019_c10`
    - was: The response must explicitly state that the binomial distribution is an \textbf{approximation} to the hypergeometric distribution due to \( n \ll N \). Failure to name the original...
    - now: The response must state that the binomial distribution is an approximation to the hypergeometric distribution because \( n \ll N \).
- **soften** `tb_0019_c08` -> `tb_0019_c13`
    - was: 
The response must label all probabilities, variables, and constants using consistent notation and, where appropriate, include units (e.g., “employees”, “%”, or “trials”) in the in...
    - now: The response should use consistent notation for variables and probabilities (e.g., X, p, n, P(X≥2)) and include appropriate units in its final interpretation. Minor informal restatements are acceptable as long as the meaning remains unambiguous.

### tb_0020
- **keep** `tb_0020_c02` -> `tb_0020_c02`
- **split** `tb_0020_c07` -> `tb_0020_c07`
    - was: The response must demonstrate that the final ratio of offspring color resulting from this genetic model matches the ratio in the original question. It should at least point out the...
    - now: The response must show that the offspring color ratio produced by the genetic model matches the ratio in the original question.
- **split** `tb_0020_c07` -> `tb_0020_c08`
    - was: The response must demonstrate that the final ratio of offspring color resulting from this genetic model matches the ratio in the original question. It should at least point out the...
    - now: The response must note the 1:2:1 segregation at gene A (black : gray : brown).
- **split** `tb_0020_c07` -> `tb_0020_c09`
    - was: The response must demonstrate that the final ratio of offspring color resulting from this genetic model matches the ratio in the original question. It should at least point out the...
    - now: The response must note the 3:1 segregation at gene B (pigmented : white).
- **format_consolidated** `tb_0020_c09` -> `tb_0020_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0021
- **keep** `tb_0021_c02` -> `tb_0021_c02`
- **split** `tb_0021_c03` -> `tb_0021_c03`
    - was: The model must provide a relatable example like AABC just to show how skipping will prevent incorrect repeated swaps. (The example must illustrate how the same letter could be swap...
    - now: The response must provide a relatable example (e.g., 'AABC') showing how skipping prevents incorrect repeated swaps.
- **split** `tb_0021_c03` -> `tb_0021_c04`
    - was: The model must provide a relatable example like AABC just to show how skipping will prevent incorrect repeated swaps. (The example must illustrate how the same letter could be swap...
    - now: The example must illustrate that, without skipping two letters after a swap, the same letter could be swapped multiple times.

### tb_0022
- **keep** `tb_0022_c02` -> `tb_0022_c02`
- **keep** `tb_0022_c04` -> `tb_0022_c04`
- **keep** `tb_0022_c07` -> `tb_0022_c07`
- **format_consolidated** `tb_0022_c08` -> `tb_0022_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0023
- **format_consolidated** `tb_0023_c07,tb_0023_c08` -> `tb_0023_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0024
- **format_consolidated** `tb_0024_c07` -> `tb_0024_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0027
- **keep** `tb_0027_c02` -> `tb_0027_c02`
- **keep** `tb_0027_c03` -> `tb_0027_c03`

### tb_0028
- **keep** `tb_0028_c02` -> `tb_0028_c02`
- **split** `tb_0028_c06` -> `tb_0028_c06`
    - was: The response must state that you have inverted the order of the substances in your expression of the mass ratio, and the correct ratio given in the problem is: (amount of Pb-206 fo...
    - now: The response must identify that the student inverted the order of the substances in the mass ratio.
- **split** `tb_0028_c06` -> `tb_0028_c07`
    - was: The response must state that you have inverted the order of the substances in your expression of the mass ratio, and the correct ratio given in the problem is: (amount of Pb-206 fo...
    - now: The response must state the correct ratio: (amount of Pb-206 formed) / (U-238 remaining) = 7.
- **keep** `tb_0028_c07` -> `tb_0028_c08`
- **split** `tb_0028_c09` -> `tb_0028_c10`
    - was: The response must identify the student's error in the final answer: the question asks "What is the value of P?", the answer should be P = 143. While 1.43×10^10 years is the correct...
    - now: The response must identify that the student reported the age (1.43×10^10 years) rather than the requested value of P.
- **split** `tb_0028_c09` -> `tb_0028_c11`
    - was: The response must identify the student's error in the final answer: the question asks "What is the value of P?", the answer should be P = 143. While 1.43×10^10 years is the correct...
    - now: The response must state that the correct value is P = 143.

### tb_0030
- **split** `tb_0030_c07` -> `tb_0030_c07`
    - was: The response must acknowledge the student’s missing background understanding, specifically, the misconception that the regression equation should yield exact values for each case o...
    - now: The response must acknowledge the student's misconception that the regression equation should yield exact values for each case.
- **split** `tb_0030_c07` -> `tb_0030_c08`
    - was: The response must acknowledge the student’s missing background understanding, specifically, the misconception that the regression equation should yield exact values for each case o...
    - now: The response must acknowledge the student's misconception that predicted values should preserve the sample mean.

### tb_0032
- **keep** `tb_0032_c01` -> `tb_0032_c01`
- **format_consolidated** `tb_0032_c05` -> `tb_0032_c07`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0034
- **keep** `tb_0034_c03` -> `tb_0034_c03`
- **keep** `tb_0034_c04` -> `tb_0034_c04`
- **keep** `tb_0034_c05` -> `tb_0034_c05`
- **keep** `tb_0034_c07` -> `tb_0034_c07`
- **keep** `tb_0034_c08` -> `tb_0034_c08`
- **format_consolidated** `tb_0034_c11` -> `tb_0034_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0035
- **format_consolidated** `tb_0035_c06` -> `tb_0035_c06`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0037
- **keep** `tb_0037_c03` -> `tb_0037_c03`
- **format_consolidated** `tb_0037_c05,tb_0037_c06` -> `tb_0037_c05`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0038
- **format_consolidated** `tb_0038_c06` -> `tb_0038_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0039
- **keep** `tb_0039_c02` -> `tb_0039_c02`

### tb_0040
- **split** `tb_0040_c03` -> `tb_0040_c03`
    - was: The response must identify and explain both convection and radiation as the key heat loss mechanisms for the hanging orb. 
    - now: The response must identify and explain convection as a key heat-loss mechanism for the hanging orb.
- **split** `tb_0040_c03` -> `tb_0040_c04`
    - was: The response must identify and explain both convection and radiation as the key heat loss mechanisms for the hanging orb. 
    - now: The response must identify and explain radiation as a key heat-loss mechanism for the hanging orb.
- **format_consolidated** `tb_0040_c06` -> `tb_0040_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0042
- **keep** `tb_0042_c01` -> `tb_0042_c01`
- **split** `tb_0042_c03` -> `tb_0042_c03`
    - was: The response must identify that H's oxidation number decreases from +1 in CH_3OH to 0 in H_2, and state that a decrease in oxidation number  = reduction.
    - now: The response must identify that H's oxidation number decreases from +1 in CH_3OH to 0 in H_2.
- **split** `tb_0042_c03` -> `tb_0042_c04`
    - was: The response must identify that H's oxidation number decreases from +1 in CH_3OH to 0 in H_2, and state that a decrease in oxidation number  = reduction.
    - now: The response must state that a decrease in oxidation number corresponds to reduction.

### tb_0043
- **format_consolidated** `tb_0043_c08` -> `tb_0043_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0044
- **keep** `tb_0044_c04` -> `tb_0044_c04`

### tb_0045
- **split** `tb_0045_c09` -> `tb_0045_c09`
    - was: The model must explain that the lagging strand has one primer per Okazaki fragement and that polymerase must keep hopping 'backwards' as a new template is exposed. 
    - now: The response must explain that the lagging strand has one primer per Okazaki fragment.
- **split** `tb_0045_c09` -> `tb_0045_c10`
    - was: The model must explain that the lagging strand has one primer per Okazaki fragement and that polymerase must keep hopping 'backwards' as a new template is exposed. 
    - now: The response must explain that the polymerase must keep hopping 'backwards' as each new template segment is exposed.

### tb_0046
- **format_consolidated** `tb_0046_c06` -> `tb_0046_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0049
- **keep** `tb_0049_c01` -> `tb_0049_c01`
- **keep** `tb_0049_c04` -> `tb_0049_c04`

### tb_0050
- **keep** `tb_0050_c03` -> `tb_0050_c03`

### tb_0051
- **format_consolidated** `tb_0051_c05` -> `tb_0051_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0052
- **split** `tb_0052_c02` -> `tb_0052_c02`
    - was: The response must state the equation C₆H₅COOH + NaOH → C₆H₅COONa + H₂O AND explain strong base forces complete reaction.
    - now: The response must present the neutralization equation for benzoic acid with NaOH (e.g., C₆H₅COOH + NaOH → C₆H₅COONa + H₂O), or an equivalent correct representation.
- **split** `tb_0052_c02` -> `tb_0052_c03`
    - was: The response must state the equation C₆H₅COOH + NaOH → C₆H₅COONa + H₂O AND explain strong base forces complete reaction.
    - now: The response must explain that the strong base (NaOH) drives the neutralization to completion.
- **split** `tb_0052_c03` -> `tb_0052_c04`
    - was: The response must explicitly state "this is a stoichiometric calculation" AND "ICE tables are not needed" AND explain why.
    - now: The response must state that this is a stoichiometric calculation and that ICE tables are not needed.
- **split** `tb_0052_c03` -> `tb_0052_c05`
    - was: The response must explicitly state "this is a stoichiometric calculation" AND "ICE tables are not needed" AND explain why.
    - now: The response must explain why ICE tables are not needed for this problem.
- **keep** `tb_0052_c09` -> `tb_0052_c11`

### tb_0053
- **format_consolidated** `tb_0053_c09` -> `tb_0053_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0054
- **format_consolidated** `tb_0054_c10` -> `tb_0054_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0055
- **keep** `tb_0055_c06` -> `tb_0055_c06`
- **format_consolidated** `tb_0055_c08,tb_0055_c10` -> `tb_0055_c10`
    - was: 2 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0056
- **split** `tb_0056_c02` -> `tb_0056_c02`
    - was: The model must identify and correct the student’s incorrect reasoning about "adding the terms" instead of subtracting.
    - now: The response must identify the student's incorrect reasoning of adding the terms instead of subtracting.
- **split** `tb_0056_c02` -> `tb_0056_c03`
    - was: The model must identify and correct the student’s incorrect reasoning about "adding the terms" instead of subtracting.
    - now: The response must correct this by explaining that the terms should be subtracted.
- **format_consolidated** `tb_0056_c07` -> `tb_0056_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0057
- **format_consolidated** `tb_0057_c09` -> `tb_0057_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0059
- **split** `tb_0059_c03` -> `tb_0059_c03`
    - was: The response must identify and correct the student’s error believing the presence of NaOH always results in a basic solution.	
    - now: The response must identify the student's error in believing that the presence of NaOH always results in a basic solution.
- **split** `tb_0059_c03` -> `tb_0059_c04`
    - was: The response must identify and correct the student’s error believing the presence of NaOH always results in a basic solution.	
    - now: The response must correct this by explaining why a solution containing NaOH is not necessarily basic.
- **format_consolidated** `tb_0059_c07` -> `tb_0059_c11`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0061
- **format_consolidated** `tb_0061_c07` -> `tb_0061_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0062
- **keep** `tb_0062_c04` -> `tb_0062_c04`

### tb_0064
- **keep** `tb_0064_c03` -> `tb_0064_c03`

### tb_0066
- **keep** `tb_0066_c06` -> `tb_0066_c06`

### tb_0067
- **keep** `tb_0067_c03` -> `tb_0067_c03`
- **format_consolidated** `tb_0067_c09` -> `tb_0067_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0068
- **keep** `tb_0068_c02` -> `tb_0068_c02`
- **format_consolidated** `tb_0068_c07` -> `tb_0068_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0070
- **format_consolidated** `tb_0070_c08` -> `tb_0070_c09`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0071
- **format_consolidated** `tb_0071_c06,tb_0071_c07` -> `tb_0071_c06`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0072
- **keep** `tb_0072_c02` -> `tb_0072_c02`
- **keep** `tb_0072_c03` -> `tb_0072_c03`
- **keep** `tb_0072_c09` -> `tb_0072_c09`

### tb_0075
- **format_consolidated** `tb_0075_c07,tb_0075_c09` -> `tb_0075_c08`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0077
- **keep** `tb_0077_c02` -> `tb_0077_c02`
- **keep** `tb_0077_c03` -> `tb_0077_c03`
- **keep** `tb_0077_c05` -> `tb_0077_c05`

### tb_0078
- **keep** `tb_0078_c08` -> `tb_0078_c08`

### tb_0079
- **split** `tb_0079_c02` -> `tb_0079_c02`
    - was: The response must provide explanations to answer the student's question, i.e. your answer is incorrect because you get a wrong circle equation, if the coordinate on y-axis is (0, b...
    - now: The response must identify the first error: the circle equation is wrong; with a y-axis point (0, b) it should be x^2 + y^2 - 2by = 0.
- **split** `tb_0079_c02` -> `tb_0079_c03`
    - was: The response must provide explanations to answer the student's question, i.e. your answer is incorrect because you get a wrong circle equation, if the coordinate on y-axis is (0, b...
    - now: The response must identify the second error: the student differentiated without applying the chain rule, giving the incorrect 2x + 2y - b\frac{dy}{dx} = 0.
- **keep** `tb_0079_c03` -> `tb_0079_c04`
- **keep** `tb_0079_c05` -> `tb_0079_c06`
- **keep** `tb_0079_c07` -> `tb_0079_c08`
- **keep** `tb_0079_c09` -> `tb_0079_c10`
- **format_consolidated** `tb_0079_c11` -> `tb_0079_c12`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0080
- **keep** `tb_0080_c10` -> `tb_0080_c10`

### tb_0081
- **keep** `tb_0081_c05` -> `tb_0081_c05`
- **format_consolidated** `tb_0081_c12` -> `tb_0081_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0084
- **format_consolidated** `tb_0084_c11` -> `tb_0084_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0086
- **format_consolidated** `tb_0086_c07` -> `tb_0086_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0087
- **keep** `tb_0087_c01` -> `tb_0087_c01`
- **format_consolidated** `tb_0087_c05` -> `tb_0087_c07`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0088
- **keep** `tb_0088_c08` -> `tb_0088_c08`

### tb_0089
- **keep** `tb_0089_c02` -> `tb_0089_c02`
- **keep** `tb_0089_c07` -> `tb_0089_c07`

### tb_0091
- **format_consolidated** `tb_0091_c07` -> `tb_0091_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0092
- **keep** `tb_0092_c04` -> `tb_0092_c04`
- **keep** `tb_0092_c06` -> `tb_0092_c06`
- **format_consolidated** `tb_0092_c08` -> `tb_0092_c09`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0095
- **format_consolidated** `tb_0095_c07` -> `tb_0095_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0096
- **keep** `tb_0096_c01` -> `tb_0096_c01`
- **format_consolidated** `tb_0096_c06` -> `tb_0096_c07`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0098
- **keep** `tb_0098_c08` -> `tb_0098_c07`
- **keep** `tb_0098_c10` -> `tb_0098_c09`
- **format_consolidated** `tb_0098_c04` -> `tb_0098_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0099
- **format_consolidated** `tb_0099_c11` -> `tb_0099_c13`
    - was: 1 orphan(s) -> aspects=['code']
    - now: The response should follow tutoring presentation conventions: present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0100
- **keep** `tb_0100_c01` -> `tb_0100_c01`

### tb_0101
- **keep** `tb_0101_c02` -> `tb_0101_c02`
- **keep** `tb_0101_c06` -> `tb_0101_c06`

### tb_0103
- **format_consolidated** `tb_0103_c11` -> `tb_0103_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0104
- **keep** `tb_0104_c02` -> `tb_0104_c02`
- **split** `tb_0104_c03` -> `tb_0104_c03`
    - was: The response must provide the correct answer to the question. Let’s go step-by-step:
Identify the ratio
[base]/[acid] = 0.150 mol / 0.250 mol = 0.600
Take the log (base-10):
log(0....
    - now: The response must set up the base/acid ratio correctly ([base]/[acid] = 0.150/0.250 = 0.600).
- **split** `tb_0104_c03` -> `tb_0104_c04`
    - was: The response must provide the correct answer to the question. Let’s go step-by-step:
Identify the ratio
[base]/[acid] = 0.150 mol / 0.250 mol = 0.600
Take the log (base-10):
log(0....
    - now: The response must apply the Henderson–Hasselbalch equation, pH = pKa + log([base]/[acid]).
- **split** `tb_0104_c03` -> `tb_0104_c05`
    - was: The response must provide the correct answer to the question. Let’s go step-by-step:
Identify the ratio
[base]/[acid] = 0.150 mol / 0.250 mol = 0.600
Take the log (base-10):
log(0....
    - now: The response must arrive at the correct pH (pH ≈ 4.54), accepting minor rounding.
- **keep** `tb_0104_c13` -> `tb_0104_c15`
- **keep** `tb_0104_c15` -> `tb_0104_c17`

### tb_0105
- **keep** `tb_0105_c07` -> `tb_0105_c07`
- **format_consolidated** `tb_0105_c19` -> `tb_0105_c20`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0106
- **format_consolidated** `tb_0106_c10` -> `tb_0106_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0107
- **keep** `tb_0107_c05` -> `tb_0107_c04`
- **format_consolidated** `tb_0107_c01` -> `tb_0107_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0108
- **format_consolidated** `tb_0108_c09` -> `tb_0108_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0110
- **format_consolidated** `tb_0110_c12` -> `tb_0110_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0111
- **keep** `tb_0111_c07` -> `tb_0111_c06`
- **format_consolidated** `tb_0111_c06` -> `tb_0111_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0112
- **keep** `tb_0112_c04` -> `tb_0112_c04`
- **keep** `tb_0112_c05` -> `tb_0112_c05`

### tb_0113
- **format_consolidated** `tb_0113_c07` -> `tb_0113_c07`
    - was: 1 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0115
- **keep** `tb_0115_c02` -> `tb_0115_c02`
- **keep** `tb_0115_c04` -> `tb_0115_c04`
- **format_consolidated** `tb_0115_c07` -> `tb_0115_c08`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0117
- **keep** `tb_0117_c05` -> `tb_0117_c05`

### tb_0118
- **keep** `tb_0118_c03` -> `tb_0118_c03`
- **keep** `tb_0118_c07` -> `tb_0118_c07`

### tb_0120
- **keep** `tb_0120_c11` -> `tb_0120_c10`
- **format_consolidated** `tb_0120_c10` -> `tb_0120_c14`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0121
- **keep** `tb_0121_c06` -> `tb_0121_c06`
- **format_consolidated** `tb_0121_c09` -> `tb_0121_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0122
- **keep** `tb_0122_c06` -> `tb_0122_c06`
- **keep** `tb_0122_c08` -> `tb_0122_c08`
- **keep** `tb_0122_c10` -> `tb_0122_c10`

### tb_0124
- **format_consolidated** `tb_0124_c11` -> `tb_0124_c15`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0125
- **keep** `tb_0125_c03` -> `tb_0125_c03`
- **keep** `tb_0125_c06` -> `tb_0125_c06`
- **format_consolidated** `tb_0125_c07` -> `tb_0125_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0126
- **keep** `tb_0126_c04` -> `tb_0126_c04`

### tb_0127
- **keep** `tb_0127_c21` -> `tb_0127_c21`
- **format_consolidated** `tb_0127_c23` -> `tb_0127_c23`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0128
- **keep** `tb_0128_c03` -> `tb_0128_c03`
- **format_consolidated** `tb_0128_c08` -> `tb_0128_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0132
- **format_consolidated** `tb_0132_c13` -> `tb_0132_c13`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0133
- **keep** `tb_0133_c01` -> `tb_0133_c01`
- **keep** `tb_0133_c02` -> `tb_0133_c02`

### tb_0134
- **keep** `tb_0134_c03` -> `tb_0134_c03`

### tb_0135
- **keep** `tb_0135_c02` -> `tb_0135_c02`
- **keep** `tb_0135_c04` -> `tb_0135_c04`
- **keep** `tb_0135_c05` -> `tb_0135_c05`
- **keep** `tb_0135_c06` -> `tb_0135_c06`
- **format_consolidated** `tb_0135_c09` -> `tb_0135_c10`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0136
- **keep** `tb_0136_c01` -> `tb_0136_c01`

### tb_0138
- **keep** `tb_0138_c06` -> `tb_0138_c06`
- **keep** `tb_0138_c07` -> `tb_0138_c07`

### tb_0140
- **keep** `tb_0140_c02` -> `tb_0140_c02`

### tb_0141
- **split** `tb_0141_c03` -> `tb_0141_c03`
    - was: The response must identify that the student's code returns -3 for [-5,-2] because it is just performing truncation (not rounding), and explain that this is wrong because standard r...
    - now: The response must identify that the student's code returns -3 for [-5,-2] because it performs truncation rather than rounding.
- **split** `tb_0141_c03` -> `tb_0141_c04`
    - was: The response must identify that the student's code returns -3 for [-5,-2] because it is just performing truncation (not rounding), and explain that this is wrong because standard r...
    - now: The response must explain that this is incorrect because standard rounding of -3.5 should give -4 (rounding away from zero / IEEE-754).
- **keep** `tb_0141_c05` -> `tb_0141_c06`

### tb_0142
- **split** `tb_0142_c05` -> `tb_0142_c05`
    - was: The response must identify the error in the original calculation and provide the fully corrected formula: 
\[\mu = \left( \frac{v^2}{r} - g \tan\theta \right) \left( \frac{\cos\the...
    - now: The response must identify the error in the student's original calculation.
- **split** `tb_0142_c05` -> `tb_0142_c06`
    - was: The response must identify the error in the original calculation and provide the fully corrected formula: 
\[\mu = \left( \frac{v^2}{r} - g \tan\theta \right) \left( \frac{\cos\the...
    - now: The response must provide the fully corrected formula: \(\mu = \left( \frac{v^2}{r} - g \tan\theta \right)\left( \frac{\cos\theta}{g} \right)\).
- **format_consolidated** `tb_0142_c08` -> `tb_0142_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0143
- **format_consolidated** `tb_0143_c07` -> `tb_0143_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0144
- **keep** `tb_0144_c05` -> `tb_0144_c05`
- **format_consolidated** `tb_0144_c08` -> `tb_0144_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0145
- **format_consolidated** `tb_0145_c09` -> `tb_0145_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0148
- **format_consolidated** `tb_0148_c05,tb_0148_c06` -> `tb_0148_c07`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0150
- **keep** `tb_0150_c01` -> `tb_0150_c01`

### tb_0151
- **keep** `tb_0151_c01` -> `tb_0151_c01`
- **format_consolidated** `tb_0151_c04,tb_0151_c05,tb_0151_c06` -> `tb_0151_c06`
    - was: 3 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0152
- **keep** `tb_0152_c05` -> `tb_0152_c05`
- **format_consolidated** `tb_0152_c09` -> `tb_0152_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0153
- **keep** `tb_0153_c22` -> `tb_0153_c22`
- **format_consolidated** `tb_0153_c28` -> `tb_0153_c28`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0154
- **keep** `tb_0154_c04` -> `tb_0154_c04`
- **format_consolidated** `tb_0154_c08` -> `tb_0154_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0155
- **keep** `tb_0155_c05` -> `tb_0155_c05`
- **format_consolidated** `tb_0155_c08` -> `tb_0155_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0157
- **keep** `tb_0157_c03` -> `tb_0157_c03`

### tb_0160
- **keep** `tb_0160_c03` -> `tb_0160_c03`
- **format_consolidated** `tb_0160_c07` -> `tb_0160_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0162
- **format_consolidated** `tb_0162_c05` -> `tb_0162_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0163
- **format_consolidated** `tb_0163_c05` -> `tb_0163_c05`
    - was: 1 orphan(s) -> aspects=['generic']
    - now: The response should follow tutoring presentation conventions: be clearly and readably presented. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0164
- **keep** `tb_0164_c05` -> `tb_0164_c05`
- **format_consolidated** `tb_0164_c08,tb_0164_c09` -> `tb_0164_c11`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0165
- **format_consolidated** `tb_0165_c13` -> `tb_0165_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0166
- **keep** `tb_0166_c02` -> `tb_0166_c02`
- **keep** `tb_0166_c14` -> `tb_0166_c14`

### tb_0167
- **keep** `tb_0167_c02` -> `tb_0167_c02`

### tb_0168
- **split** `tb_0168_c02` -> `tb_0168_c02`
    - was: The response must identify and explain the background knowledge necessary for the problem - definition of slope (how much Y changes for each unit change in X); definition of interc...
    - now: The response must define slope (how much Y changes for each unit change in X).
- **split** `tb_0168_c02` -> `tb_0168_c03`
    - was: The response must identify and explain the background knowledge necessary for the problem - definition of slope (how much Y changes for each unit change in X); definition of interc...
    - now: The response must define the intercept (b = \bar{Y} - a\bar{x}, chosen so the line best fits the data).
- **keep** `tb_0168_c03` -> `tb_0168_c04`

### tb_0169
- **keep** `tb_0169_c01` -> `tb_0169_c01`
- **keep** `tb_0169_c13` -> `tb_0169_c13`
- **keep** `tb_0169_c14` -> `tb_0169_c14`
- **format_consolidated** `tb_0169_c16` -> `tb_0169_c17`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0171
- **keep** `tb_0171_c02` -> `tb_0171_c02`
- **keep** `tb_0171_c05` -> `tb_0171_c05`
- **soften** `tb_0171_c08` -> `tb_0171_c08`
    - was: The model response should be easily readable and clearly formatted with bullet points, headers, tables, graphs and mathematical expressions whenever necessary. If any of these are ...
    - now: The model response should be easily readable and clearly formatted, using bullet points, headers, tables, graphs, and mathematical expressions where they aid clarity.
- **format_consolidated** `tb_0171_c08` -> `tb_0171_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0173
- **keep** `tb_0173_c08` -> `tb_0173_c07`
- **format_consolidated** `tb_0173_c05` -> `tb_0173_c08`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0176
- **split** `tb_0176_c05` -> `tb_0176_c05`
    - was: The model must answer the student's question about what would change if the car starts to brake by explaining that braking introduces a new (tangential) braking force into the prob...
    - now: The model must answer the student's question about what would change if the car starts to brake by explaining that braking introduces a new (tangential) braking force into the problem, which results in a total applied force given by $F_{total}=\sqrt{F_{radial}^2+F_{tangential}^2}$.
- **split** `tb_0176_c05` -> `tb_0176_c06`
    - was: The model must answer the student's question about what would change if the car starts to brake by explaining that braking introduces a new (tangential) braking force into the prob...
    - now: The model must explain that, once the car starts to brake, the minimum friction coefficient must now be sufficient to produce a friction force that is greater or equal than the resulting force from the combination of braking and the centripetal force.

### tb_0177
- **split** `tb_0177_c03` -> `tb_0177_c03`
    - was: The response must mention that a typical titration curve of a weak acid with a strong base shows a gradual increase in pH as the base is added, followed by a sharp rise at the equi...
    - now: The response must mention that a typical titration curve of a weak acid with a strong base shows a gradual increase in pH as the base is added, followed by a sharp rise at the equivalence point.
- **split** `tb_0177_c03` -> `tb_0177_c04`
    - was: The response must mention that a typical titration curve of a weak acid with a strong base shows a gradual increase in pH as the base is added, followed by a sharp rise at the equi...
    - now: The response must mention that the half-equivalence point is where half of the acid has been neutralized, and pH = pKa.

### tb_0178
- **keep** `tb_0178_c04` -> `tb_0178_c04`
- **keep** `tb_0178_c05` -> `tb_0178_c05`
- **format_consolidated** `tb_0178_c06` -> `tb_0178_c06`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0180
- **keep** `tb_0180_c01` -> `tb_0180_c01`

### tb_0181
- **keep** `tb_0181_c07` -> `tb_0181_c07`

### tb_0182
- **keep** `tb_0182_c03` -> `tb_0182_c02`
- **keep** `tb_0182_c05` -> `tb_0182_c04`
- **keep** `tb_0182_c08` -> `tb_0182_c07`
- **keep** `tb_0182_c09` -> `tb_0182_c08`
- **format_consolidated** `tb_0182_c01` -> `tb_0182_c11`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0183
- **keep** `tb_0183_c01` -> `tb_0183_c01`

### tb_0186
- **keep** `tb_0186_c02` -> `tb_0186_c02`

### tb_0187
- **soften** `tb_0187_c01` -> `tb_0187_c01`
    - was: The response must acknowledge the student's expressed they are confused about how the treatments are actually assigned to each twin similar to: "I can understand how the assignment...
    - now: The response must acknowledge that the student expressed confusion about how the treatments are actually assigned to each twin.

### tb_0188
- **keep** `tb_0188_c07` -> `tb_0188_c07`

### tb_0189
- **keep** `tb_0189_c03` -> `tb_0189_c02`
- **keep** `tb_0189_c05` -> `tb_0189_c04`
- **keep** `tb_0189_c06` -> `tb_0189_c05`
- **keep** `tb_0189_c08` -> `tb_0189_c07`
- **format_consolidated** `tb_0189_c01` -> `tb_0189_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0195
- **keep** `tb_0195_c06` -> `tb_0195_c06`

### tb_0196
- **keep** `tb_0196_c03` -> `tb_0196_c03`
- **format_consolidated** `tb_0196_c07` -> `tb_0196_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0197
- **keep** `tb_0197_c05` -> `tb_0197_c05`
- **format_consolidated** `tb_0197_c12` -> `tb_0197_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0199
- **format_consolidated** `tb_0199_c09` -> `tb_0199_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0201
- **keep** `tb_0201_c06` -> `tb_0201_c06`

### tb_0202
- **format_consolidated** `tb_0202_c01` -> `tb_0202_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0204
- **keep** `tb_0204_c05` -> `tb_0204_c05`
- **format_consolidated** `tb_0204_c09` -> `tb_0204_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0209
- **keep** `tb_0209_c05` -> `tb_0209_c05`
- **format_consolidated** `tb_0209_c08,tb_0209_c09` -> `tb_0209_c09`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0211
- **keep** `tb_0211_c04` -> `tb_0211_c04`
- **format_consolidated** `tb_0211_c06` -> `tb_0211_c07`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0212
- **keep** `tb_0212_c05` -> `tb_0212_c05`
- **keep** `tb_0212_c06` -> `tb_0212_c06`

### tb_0213
- **format_consolidated** `tb_0213_c10` -> `tb_0213_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0214
- **keep** `tb_0214_c05` -> `tb_0214_c05`
- **format_consolidated** `tb_0214_c08` -> `tb_0214_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0217
- **format_consolidated** `tb_0217_c05` -> `tb_0217_c05`
    - was: 1 orphan(s) -> aspects=['code']
    - now: The response should follow tutoring presentation conventions: present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0219
- **keep** `tb_0219_c02` -> `tb_0219_c02`

### tb_0220
- **keep** `tb_0220_c02` -> `tb_0220_c02`

### tb_0221
- **format_consolidated** `tb_0221_c01,tb_0221_c09` -> `tb_0221_c08`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0222
- **keep** `tb_0222_c03` -> `tb_0222_c03`
- **format_consolidated** `tb_0222_c07` -> `tb_0222_c09`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0223
- **format_consolidated** `tb_0223_c07` -> `tb_0223_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0224
- **keep** `tb_0224_c03` -> `tb_0224_c03`
- **format_consolidated** `tb_0224_c10,tb_0224_c11` -> `tb_0224_c11`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0225
- **keep** `tb_0225_c03` -> `tb_0225_c03`
- **keep** `tb_0225_c07` -> `tb_0225_c07`
- **format_consolidated** `tb_0225_c11` -> `tb_0225_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0226
- **keep** `tb_0226_c13` -> `tb_0226_c13`
- **keep** `tb_0226_c18` -> `tb_0226_c18`

### tb_0227
- **format_consolidated** `tb_0227_c06` -> `tb_0227_c07`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0229
- **format_consolidated** `tb_0229_c09` -> `tb_0229_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0231
- **keep** `tb_0231_c08` -> `tb_0231_c08`

### tb_0232
- **format_consolidated** `tb_0232_c08` -> `tb_0232_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0233
- **keep** `tb_0233_c05` -> `tb_0233_c05`
- **keep** `tb_0233_c07` -> `tb_0233_c07`

### tb_0234
- **keep** `tb_0234_c04` -> `tb_0234_c04`
- **split** `tb_0234_c05` -> `tb_0234_c05`
    - was: The response must explain how to find both a relative minimum and relative maximum (since the student is wondering about both) by describing how H'(t) behaves when a value is plugg...
    - now: The response must explain that a relative maximum occurs where H'(t) transitions from positive to negative (a value below the critical point gives a positive H'(t), a value above gives a negative H'(t)).
- **split** `tb_0234_c05` -> `tb_0234_c06`
    - was: The response must explain how to find both a relative minimum and relative maximum (since the student is wondering about both) by describing how H'(t) behaves when a value is plugg...
    - now: The response must explain that a relative minimum occurs with the inverse sign pattern (H'(t) transitions from negative to positive).
- **format_consolidated** `tb_0234_c08` -> `tb_0234_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0235
- **format_consolidated** `tb_0235_c07` -> `tb_0235_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0236
- **keep** `tb_0236_c07` -> `tb_0236_c07`

### tb_0238
- **format_consolidated** `tb_0238_c18` -> `tb_0238_c18`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0240
- **keep** `tb_0240_c04` -> `tb_0240_c04`
- **keep** `tb_0240_c05` -> `tb_0240_c05`

### tb_0241
- **keep** `tb_0241_c03` -> `tb_0241_c03`

### tb_0242
- **keep** `tb_0242_c04` -> `tb_0242_c04`

### tb_0243
- **keep** `tb_0243_c11` -> `tb_0243_c11`
- **keep** `tb_0243_c14` -> `tb_0243_c14`
- **format_consolidated** `tb_0243_c15` -> `tb_0243_c16`
    - was: 1 orphan(s) -> aspects=['generic']
    - now: The response should follow tutoring presentation conventions: be clearly and readably presented. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0244
- **keep** `tb_0244_c04` -> `tb_0244_c04`
- **keep** `tb_0244_c07` -> `tb_0244_c07`
- **format_consolidated** `tb_0244_c10,tb_0244_c11` -> `tb_0244_c10`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0245
- **keep** `tb_0245_c03` -> `tb_0245_c03`
- **keep** `tb_0245_c07` -> `tb_0245_c07`
- **keep** `tb_0245_c11` -> `tb_0245_c11`
- **keep** `tb_0245_c13` -> `tb_0245_c13`
- **format_consolidated** `tb_0245_c17` -> `tb_0245_c17`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0248
- **keep** `tb_0248_c01` -> `tb_0248_c01`
- **format_consolidated** `tb_0248_c07` -> `tb_0248_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0251
- **format_consolidated** `tb_0251_c08` -> `tb_0251_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'math', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0254
- **keep** `tb_0254_c03` -> `tb_0254_c03`
- **split** `tb_0254_c04` -> `tb_0254_c04`
    - was: The response must explain z serves two different purposes: (1) z in π(0.4z)² determines the radius/volume of water at that height, and (2) (10-z) determines the distance that water...
    - now: The response must explain that z in π(0.4z)² determines the radius/volume of water at that height.
- **split** `tb_0254_c04` -> `tb_0254_c05`
    - was: The response must explain z serves two different purposes: (1) z in π(0.4z)² determines the radius/volume of water at that height, and (2) (10-z) determines the distance that water...
    - now: The response must explain that (10-z) determines the distance that water needs to travel.

### tb_0255
- **format_consolidated** `tb_0255_c06` -> `tb_0255_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0258
- **keep** `tb_0258_c04` -> `tb_0258_c03`
- **format_consolidated** `tb_0258_c01` -> `tb_0258_c05`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0260
- **format_consolidated** `tb_0260_c06` -> `tb_0260_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0261
- **format_consolidated** `tb_0261_c10` -> `tb_0261_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0262
- **format_consolidated** `tb_0262_c09` -> `tb_0262_c10`
    - was: 1 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0263
- **split** `tb_0263_c02` -> `tb_0263_c02`
    - was: The response must identify and correct the student's assertion that time complexity is O(RxCx4), explaining that despite 4 recursive calls per cell, each cell is visited exactly on...
    - now: The response must identify that the student's assertion that the time complexity is O(RxCx4) is incorrect.
- **split** `tb_0263_c02` -> `tb_0263_c03`
    - was: The response must identify and correct the student's assertion that time complexity is O(RxCx4), explaining that despite 4 recursive calls per cell, each cell is visited exactly on...
    - now: The response must explain that despite 4 recursive calls per cell, each cell is visited exactly once due to the visited array mechanism, making the actual complexity O(RxC).
- **split** `tb_0263_c03` -> `tb_0263_c04`
    - was: The response must identify and correct the student's claim that a 1000x1000 all-land map would require 1,000,000 recursive levels, explaining that maximum recursion depth equals th...
    - now: The response must identify that the student's claim that a 1000x1000 all-land map would require 1,000,000 recursive levels is incorrect.
- **split** `tb_0263_c03` -> `tb_0263_c05`
    - was: The response must identify and correct the student's claim that a 1000x1000 all-land map would require 1,000,000 recursive levels, explaining that maximum recursion depth equals th...
    - now: The response must explain that the maximum recursion depth equals the longest connected path which is bounded by approximately R+C-1 (roughly 1999 for a 1000x1000 grid), not the total number of cells (RxC=1,000,000).

### tb_0264
- **format_consolidated** `tb_0264_c08` -> `tb_0264_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0265
- **keep** `tb_0265_c03` -> `tb_0265_c03`
- **format_consolidated** `tb_0265_c07` -> `tb_0265_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0266
- **format_consolidated** `tb_0266_c06` -> `tb_0266_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0268
- **keep** `tb_0268_c12` -> `tb_0268_c12`
- **format_consolidated** `tb_0268_c13` -> `tb_0268_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0269
- **keep** `tb_0269_c01` -> `tb_0269_c01`
- **keep** `tb_0269_c02` -> `tb_0269_c02`
- **keep** `tb_0269_c04` -> `tb_0269_c04`

### tb_0270
- **format_consolidated** `tb_0270_c05` -> `tb_0270_c06`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0271
- **split** `tb_0271_c03` -> `tb_0271_c03`
    - was: The response must identify and explain all four bugs in the student's code: (1) array dimensions should be [m+1][n+1] not [m][n], (2) dp[0][0] should equal true not a character com...
    - now: The response must identify and explain the bug that the array dimensions should be [m+1][n+1] not [m][n].
- **split** `tb_0271_c03` -> `tb_0271_c04`
    - was: The response must identify and explain all four bugs in the student's code: (1) array dimensions should be [m+1][n+1] not [m][n], (2) dp[0][0] should equal true not a character com...
    - now: The response must identify and explain the bug that dp[0][0] should equal true, not a character comparison.
- **split** `tb_0271_c03` -> `tb_0271_c05`
    - was: The response must identify and explain all four bugs in the student's code: (1) array dimensions should be [m+1][n+1] not [m][n], (2) dp[0][0] should equal true not a character com...
    - now: The response must identify and explain the bug that character access should use charAt(i-1) not charAt(i) when at dp[i][j].
- **split** `tb_0271_c03` -> `tb_0271_c06`
    - was: The response must identify and explain all four bugs in the student's code: (1) array dimensions should be [m+1][n+1] not [m][n], (2) dp[0][0] should equal true not a character com...
    - now: The response must identify and explain the bug that the second if statement overwrites the first result instead of combining with the OR operator.
- **format_consolidated** `tb_0271_c10` -> `tb_0271_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0273
- **keep** `tb_0273_c09` -> `tb_0273_c09`
- **format_consolidated** `tb_0273_c14` -> `tb_0273_c16`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0276
- **keep** `tb_0276_c01` -> `tb_0276_c01`
- **split** `tb_0276_c05` -> `tb_0276_c05`
    - was: The response must state that Apetamin-G amplifies the pathway because (a) extrahepatic GHR mRNA yields more GH receptors and more IGF-1 per pulse, (b) leucine directly stimulates m...
    - now: The response must state that Apetamin-G amplifies the pathway because extrahepatic GHR mRNA yields more GH receptors and more IGF-1 per pulse.
- **split** `tb_0276_c05` -> `tb_0276_c06`
    - was: The response must state that Apetamin-G amplifies the pathway because (a) extrahepatic GHR mRNA yields more GH receptors and more IGF-1 per pulse, (b) leucine directly stimulates m...
    - now: The response must state that Apetamin-G amplifies the pathway because leucine directly stimulates mTORC1.
- **split** `tb_0276_c05` -> `tb_0276_c07`
    - was: The response must state that Apetamin-G amplifies the pathway because (a) extrahepatic GHR mRNA yields more GH receptors and more IGF-1 per pulse, (b) leucine directly stimulates m...
    - now: The response must state that Apetamin-G amplifies the pathway because B vitamins supply cofactors for ATP production and amino acid metabolism.

### tb_0277
- **split** `tb_0277_c03` -> `tb_0277_c03`
    - was: The response needs to explain the solubility rules by stating that nitrates are generally (but not always) soluble, therefore Cu(NO₃)₂ must be soluble, and sulfates are generally s...
    - now: The response needs to explain the solubility rules by stating that nitrates are generally (but not always) soluble, therefore Cu(NO₃)₂ must be soluble.
- **split** `tb_0277_c03` -> `tb_0277_c04`
    - was: The response needs to explain the solubility rules by stating that nitrates are generally (but not always) soluble, therefore Cu(NO₃)₂ must be soluble, and sulfates are generally s...
    - now: The response needs to explain the solubility rules by stating that sulfates are generally soluble except for CaSO₄, BaSO₄, and PbSO₄; therefore, BaSO₄ is an exception and is insoluble.

### tb_0278
- **keep** `tb_0278_c02` -> `tb_0278_c02`
- **keep** `tb_0278_c08` -> `tb_0278_c08`

### tb_0279
- **format_consolidated** `tb_0279_c10` -> `tb_0279_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0280
- **format_consolidated** `tb_0280_c07` -> `tb_0280_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0282
- **keep** `tb_0282_c02` -> `tb_0282_c02`
- **keep** `tb_0282_c04` -> `tb_0282_c04`
- **keep** `tb_0282_c05` -> `tb_0282_c05`
- **keep** `tb_0282_c08` -> `tb_0282_c08`
- **format_consolidated** `tb_0282_c09` -> `tb_0282_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0283
- **keep** `tb_0283_c05` -> `tb_0283_c05`

### tb_0284
- **keep** `tb_0284_c01` -> `tb_0284_c01`

### tb_0285
- **keep** `tb_0285_c02` -> `tb_0285_c02`
- **keep** `tb_0285_c08` -> `tb_0285_c08`
- **format_consolidated** `tb_0285_c12` -> `tb_0285_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0289
- **keep** `tb_0289_c01` -> `tb_0289_c01`
- **keep** `tb_0289_c03` -> `tb_0289_c03`
- **keep** `tb_0289_c06` -> `tb_0289_c06`
- **keep** `tb_0289_c08` -> `tb_0289_c08`
- **format_consolidated** `tb_0289_c09` -> `tb_0289_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0290
- **keep** `tb_0290_c02` -> `tb_0290_c02`

### tb_0291
- **keep** `tb_0291_c04` -> `tb_0291_c04`
- **format_consolidated** `tb_0291_c09` -> `tb_0291_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0292
- **keep** `tb_0292_c04` -> `tb_0292_c04`

### tb_0294
- **keep** `tb_0294_c01` -> `tb_0294_c01`
- **keep** `tb_0294_c05` -> `tb_0294_c05`
- **split** `tb_0294_c07` -> `tb_0294_c07`
    - was: The response must identify that the student is wrong in claiming that the standard deviation is 1, and the response must report that the true standard deviation for this problem is...
    - now: The response must identify that the student is wrong in claiming that the standard deviation is 1.
- **split** `tb_0294_c07` -> `tb_0294_c08`
    - was: The response must identify that the student is wrong in claiming that the standard deviation is 1, and the response must report that the true standard deviation for this problem is...
    - now: The response must report that the true standard deviation for this problem is 11.99.

### tb_0295
- **format_consolidated** `tb_0295_c01` -> `tb_0295_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0296
- **split** `tb_0296_c01` -> `tb_0296_c01`
    - was: The response must explicitly state Bayes’ theorem formula ($P(D\mid T)=\frac{P(T\mid D)\times P(D)}{P(T)}$), label its components (prior, likelihood, evidence), and show how the gi...
    - now: The response must state Bayes' theorem, P(D|T) = P(T|D)·P(D) / P(T), or an equivalent formulation.
- **split** `tb_0296_c01` -> `tb_0296_c02`
    - was: The response must explicitly state Bayes’ theorem formula ($P(D\mid T)=\frac{P(T\mid D)\times P(D)}{P(T)}$), label its components (prior, likelihood, evidence), and show how the gi...
    - now: The response must identify the components of the formula (prior P(D), likelihood P(T|D), evidence P(T)).
- **split** `tb_0296_c01` -> `tb_0296_c03`
    - was: The response must explicitly state Bayes’ theorem formula ($P(D\mid T)=\frac{P(T\mid D)\times P(D)}{P(T)}$), label its components (prior, likelihood, evidence), and show how the gi...
    - now: The response must substitute the given test parameters into the formula and compute the posterior probability.

### tb_0297
- **format_consolidated** `tb_0297_c08` -> `tb_0297_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0298
- **keep** `tb_0298_c03` -> `tb_0298_c03`
- **format_consolidated** `tb_0298_c10` -> `tb_0298_c12`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0299
- **keep** `tb_0299_c06` -> `tb_0299_c05`
- **keep** `tb_0299_c09` -> `tb_0299_c08`
- **format_consolidated** `tb_0299_c03` -> `tb_0299_c10`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0300
- **keep** `tb_0300_c02` -> `tb_0300_c02`
- **keep** `tb_0300_c03` -> `tb_0300_c03`
- **keep** `tb_0300_c04` -> `tb_0300_c04`
- **format_consolidated** `tb_0300_c06,tb_0300_c07` -> `tb_0300_c06`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0301
- **keep** `tb_0301_c01` -> `tb_0301_c01`

### tb_0302
- **format_consolidated** `tb_0302_c04` -> `tb_0302_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0303
- **keep** `tb_0303_c02` -> `tb_0303_c02`
- **keep** `tb_0303_c05` -> `tb_0303_c05`
- **keep** `tb_0303_c07` -> `tb_0303_c07`

### tb_0304
- **keep** `tb_0304_c03` -> `tb_0304_c03`

### tb_0305
- **keep** `tb_0305_c03` -> `tb_0305_c03`

### tb_0306
- **format_consolidated** `tb_0306_c08` -> `tb_0306_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0307
- **keep** `tb_0307_c10` -> `tb_0307_c10`

### tb_0308
- **format_consolidated** `tb_0308_c05` -> `tb_0308_c05`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0310
- **format_consolidated** `tb_0310_c10` -> `tb_0310_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0312
- **keep** `tb_0312_c03` -> `tb_0312_c03`
- **format_consolidated** `tb_0312_c09` -> `tb_0312_c09`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0314
- **keep** `tb_0314_c03` -> `tb_0314_c03`

### tb_0316
- **keep** `tb_0316_c07` -> `tb_0316_c07`
- **format_consolidated** `tb_0316_c09` -> `tb_0316_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0317
- **format_consolidated** `tb_0317_c09` -> `tb_0317_c09`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0320
- **format_consolidated** `tb_0320_c10` -> `tb_0320_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0321
- **keep** `tb_0321_c09` -> `tb_0321_c08`
- **format_consolidated** `tb_0321_c01` -> `tb_0321_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0322
- **format_consolidated** `tb_0322_c08` -> `tb_0322_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0325
- **keep** `tb_0325_c01` -> `tb_0325_c01`
- **keep** `tb_0325_c02` -> `tb_0325_c02`
- **keep** `tb_0325_c05` -> `tb_0325_c05`

### tb_0326
- **keep** `tb_0326_c02` -> `tb_0326_c02`
- **keep** `tb_0326_c05` -> `tb_0326_c05`

### tb_0328
- **keep** `tb_0328_c04` -> `tb_0328_c04`
- **format_consolidated** `tb_0328_c06,tb_0328_c08` -> `tb_0328_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0329
- **keep** `tb_0329_c01` -> `tb_0329_c01`
- **keep** `tb_0329_c07` -> `tb_0329_c06`
- **format_consolidated** `tb_0329_c06` -> `tb_0329_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0331
- **format_consolidated** `tb_0331_c01` -> `tb_0331_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0334
- **keep** `tb_0334_c01` -> `tb_0334_c01`
- **format_consolidated** `tb_0334_c13` -> `tb_0334_c13`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0335
- **keep** `tb_0335_c01` -> `tb_0335_c01`
- **format_consolidated** `tb_0335_c08` -> `tb_0335_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0336
- **split** `tb_0336_c03` -> `tb_0336_c03`
    - was: The response must identify that the student made an arithmetic error in part c) by equating $\frac{1}{5}$ to $\frac{5}{35}$, when it should be $\frac{7}{35}$. If done correctly, th...
    - now: The response must identify that the student made an arithmetic error in part c) by equating $\frac{1}{5}$ to $\frac{5}{35}$, when it should be $\frac{7}{35}$.
- **split** `tb_0336_c03` -> `tb_0336_c04`
    - was: The response must identify that the student made an arithmetic error in part c) by equating $\frac{1}{5}$ to $\frac{5}{35}$, when it should be $\frac{7}{35}$. If done correctly, th...
    - now: The response must state that, done correctly, the answer for part c) should have been $\frac{1}{7}MR^2\omega_0^2$, not $\frac{3}{35}MR^2\omega_0^2$.
- **keep** `tb_0336_c06` -> `tb_0336_c07`
- **format_consolidated** `tb_0336_c09,tb_0336_c10` -> `tb_0336_c10`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0338
- **keep** `tb_0338_c01` -> `tb_0338_c01`
- **keep** `tb_0338_c02` -> `tb_0338_c02`
- **keep** `tb_0338_c04` -> `tb_0338_c04`
- **keep** `tb_0338_c05` -> `tb_0338_c05`
- **keep** `tb_0338_c06` -> `tb_0338_c06`
- **keep** `tb_0338_c08` -> `tb_0338_c08`

### tb_0339
- **format_consolidated** `tb_0339_c03` -> `tb_0339_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0340
- **split** `tb_0340_c08` -> `tb_0340_c08`
    - was: The response must include a validation for the student's question: "That means the prediction was 8.14 kg more than the actual weight, right?" similar to: "Totally! But does that a...
    - now: The response must acknowledge and validate the student's restatement that the prediction was 8.14 kg more than the actual weight (confirming the interpretation is correct).
- **split** `tb_0340_c08` -> `tb_0340_c09`
    - was: The response must include a validation for the student's question: "That means the prediction was 8.14 kg more than the actual weight, right?" similar to: "Totally! But does that a...
    - now: The response must redirect the student toward the actual goal of the problem (e.g., prompting them to consider what the residual means in context rather than stopping at the arithmetic).
- **format_consolidated** `tb_0340_c10` -> `tb_0340_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0341
- **format_consolidated** `tb_0341_c07` -> `tb_0341_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0342
- **keep** `tb_0342_c02` -> `tb_0342_c02`

### tb_0343
- **keep** `tb_0343_c09` -> `tb_0343_c08`
- **format_consolidated** `tb_0343_c07` -> `tb_0343_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0344
- **split** `tb_0344_c06` -> `tb_0344_c04`
    - was: The response must  identify all student's errors: 1. the student’s final probability (5929/22500) 2. the student’s use of an incorrect probability for Bob playing paper (1/15 inste...
    - now: The response must identify that the student's final probability (5929/22500) is incorrect.
- **split** `tb_0344_c06` -> `tb_0344_c05`
    - was: The response must  identify all student's errors: 1. the student’s final probability (5929/22500) 2. the student’s use of an incorrect probability for Bob playing paper (1/15 inste...
    - now: The response must identify that the student used an incorrect probability for Bob playing paper (1/15 instead of 1/5).
- **split** `tb_0344_c06` -> `tb_0344_c06`
    - was: The response must  identify all student's errors: 1. the student’s final probability (5929/22500) 2. the student’s use of an incorrect probability for Bob playing paper (1/15 inste...
    - now: The response must identify that the student undercounted the valid game-winning sequences by stating that there are two.
- **format_consolidated** `tb_0344_c01,tb_0344_c02` -> `tb_0344_c10`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0346
- **format_consolidated** `tb_0346_c09` -> `tb_0346_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0347
- **keep** `tb_0347_c03` -> `tb_0347_c03`

### tb_0350
- **keep** `tb_0350_c03` -> `tb_0350_c03`
- **keep** `tb_0350_c07` -> `tb_0350_c07`

### tb_0352
- **keep** `tb_0352_c06` -> `tb_0352_c06`

### tb_0354
- **keep** `tb_0354_c14` -> `tb_0354_c13`
- **format_consolidated** `tb_0354_c11` -> `tb_0354_c18`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0356
- **keep** `tb_0356_c01` -> `tb_0356_c01`
- **keep** `tb_0356_c02` -> `tb_0356_c02`
- **keep** `tb_0356_c03` -> `tb_0356_c03`

### tb_0357
- **keep** `tb_0357_c03` -> `tb_0357_c03`
- **keep** `tb_0357_c07` -> `tb_0357_c07`
- **keep** `tb_0357_c12` -> `tb_0357_c10`
- **keep** `tb_0357_c13` -> `tb_0357_c11`
- **format_consolidated** `tb_0357_c08,tb_0357_c10` -> `tb_0357_c17`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0358
- **format_consolidated** `tb_0358_c08` -> `tb_0358_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0359
- **keep** `tb_0359_c03` -> `tb_0359_c03`
- **format_consolidated** `tb_0359_c19` -> `tb_0359_c20`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0361
- **format_consolidated** `tb_0361_c08` -> `tb_0361_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0362
- **keep** `tb_0362_c06` -> `tb_0362_c04`
- **format_consolidated** `tb_0362_c02,tb_0362_c03` -> `tb_0362_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0363
- **keep** `tb_0363_c03` -> `tb_0363_c03`
- **keep** `tb_0363_c05` -> `tb_0363_c05`
- **keep** `tb_0363_c06` -> `tb_0363_c06`
- **keep** `tb_0363_c08` -> `tb_0363_c08`
- **keep** `tb_0363_c15` -> `tb_0363_c15`

### tb_0364
- **keep** `tb_0364_c06` -> `tb_0364_c06`
- **keep** `tb_0364_c09` -> `tb_0364_c09`
- **format_consolidated** `tb_0364_c13` -> `tb_0364_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0366
- **format_consolidated** `tb_0366_c07` -> `tb_0366_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0367
- **keep** `tb_0367_c07` -> `tb_0367_c07`

### tb_0368
- **keep** `tb_0368_c01` -> `tb_0368_c01`
- **keep** `tb_0368_c02` -> `tb_0368_c02`
- **split** `tb_0368_c03` -> `tb_0368_c03`
    - was: The response must note that the student omitted the test name and required conditions/assumptions (independence, normality/CLT) and must state that these should be checked for a va...
    - now: The response must note that the student omitted the test name (a two-sample t-test).
- **split** `tb_0368_c03` -> `tb_0368_c04`
    - was: The response must note that the student omitted the test name and required conditions/assumptions (independence, normality/CLT) and must state that these should be checked for a va...
    - now: The response must note that the student omitted the required conditions/assumptions (independence, normality/CLT) that should be checked for a valid two-sample t-test.
- **format_consolidated** `tb_0368_c06` -> `tb_0368_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0369
- **keep** `tb_0369_c17` -> `tb_0369_c17`

### tb_0370
- **keep** `tb_0370_c09` -> `tb_0370_c09`
- **format_consolidated** `tb_0370_c14,tb_0370_c15` -> `tb_0370_c14`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0371
- **keep** `tb_0371_c02` -> `tb_0371_c02`
- **keep** `tb_0371_c04` -> `tb_0371_c04`
- **keep** `tb_0371_c05` -> `tb_0371_c05`
- **keep** `tb_0371_c07` -> `tb_0371_c07`
- **format_consolidated** `tb_0371_c19` -> `tb_0371_c19`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0372
- **keep** `tb_0372_c01` -> `tb_0372_c01`
- **format_consolidated** `tb_0372_c07` -> `tb_0372_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0373
- **keep** `tb_0373_c08` -> `tb_0373_c08`

### tb_0374
- **keep** `tb_0374_c07` -> `tb_0374_c06`
- **split** `tb_0374_c08` -> `tb_0374_c07`
    - was: The response must provide the correct d.f. for the t-test: For equal variances, $\text{df} = n_1 + n_2 - 2 = 98$, and for unequal variances: Welch–Satterthwaite formula should be u...
    - now: The response must provide the correct d.f. for the equal-variances case: \(\text{df} = n_1 + n_2 - 2 = 98\).
- **split** `tb_0374_c08` -> `tb_0374_c08`
    - was: The response must provide the correct d.f. for the t-test: For equal variances, $\text{df} = n_1 + n_2 - 2 = 98$, and for unequal variances: Welch–Satterthwaite formula should be u...
    - now: The response must state that for unequal variances the Welch–Satterthwaite formula should be used to compute the d.f.
- **format_consolidated** `tb_0374_c01` -> `tb_0374_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0375
- **format_consolidated** `tb_0375_c02` -> `tb_0375_c14`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0376
- **keep** `tb_0376_c02` -> `tb_0376_c02`
- **keep** `tb_0376_c18` -> `tb_0376_c18`

### tb_0377
- **keep** `tb_0377_c01` -> `tb_0377_c01`

### tb_0378
- **keep** `tb_0378_c13` -> `tb_0378_c13`

### tb_0379
- **keep** `tb_0379_c02` -> `tb_0379_c02`
- **keep** `tb_0379_c05` -> `tb_0379_c05`
- **keep** `tb_0379_c06` -> `tb_0379_c06`
- **keep** `tb_0379_c07` -> `tb_0379_c07`
- **format_consolidated** `tb_0379_c11` -> `tb_0379_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0380
- **format_consolidated** `tb_0380_c05` -> `tb_0380_c08`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0381
- **keep** `tb_0381_c03` -> `tb_0381_c03`
- **keep** `tb_0381_c06` -> `tb_0381_c06`

### tb_0382
- **keep** `tb_0382_c07` -> `tb_0382_c06`
- **keep** `tb_0382_c08` -> `tb_0382_c07`
- **format_consolidated** `tb_0382_c06` -> `tb_0382_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0384
- **keep** `tb_0384_c01` -> `tb_0384_c01`
- **split** `tb_0384_c06` -> `tb_0384_c05`
    - was: The model must acknowledge that the student correctly noted that the father's contribution of either an X or Y chromosome determines the sex of the child and additionally, the mode...
    - now: The model must acknowledge that the student correctly noted that the father's contribution of either an X or Y chromosome determines the sex of the child.
- **split** `tb_0384_c06` -> `tb_0384_c06`
    - was: The model must acknowledge that the student correctly noted that the father's contribution of either an X or Y chromosome determines the sex of the child and additionally, the mode...
    - now: The model should acknowledge that the student is familiar with the Punnett square as a tool to explore inheritance patterns.
- **format_consolidated** `tb_0384_c04` -> `tb_0384_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0386
- **keep** `tb_0386_c04` -> `tb_0386_c04`

### tb_0388
- **keep** `tb_0388_c02` -> `tb_0388_c02`
- **keep** `tb_0388_c04` -> `tb_0388_c04`
- **format_consolidated** `tb_0388_c05` -> `tb_0388_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0389
- **format_consolidated** `tb_0389_c09` -> `tb_0389_c11`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0390
- **keep** `tb_0390_c01` -> `tb_0390_c01`
- **keep** `tb_0390_c03` -> `tb_0390_c03`
- **keep** `tb_0390_c06` -> `tb_0390_c06`

### tb_0391
- **keep** `tb_0391_c03` -> `tb_0391_c03`
- **format_consolidated** `tb_0391_c05` -> `tb_0391_c06`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0392
- **format_consolidated** `tb_0392_c06` -> `tb_0392_c06`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0394
- **keep** `tb_0394_c04` -> `tb_0394_c04`
- **keep** `tb_0394_c06` -> `tb_0394_c06`
- **keep** `tb_0394_c07` -> `tb_0394_c07`

### tb_0395
- **format_consolidated** `tb_0395_c13` -> `tb_0395_c20`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0397
- **keep** `tb_0397_c01` -> `tb_0397_c01`
- **keep** `tb_0397_c02` -> `tb_0397_c02`
- **keep** `tb_0397_c03` -> `tb_0397_c03`
- **keep** `tb_0397_c05` -> `tb_0397_c05`

### tb_0398
- **keep** `tb_0398_c02` -> `tb_0398_c02`
- **keep** `tb_0398_c06` -> `tb_0398_c06`

### tb_0399
- **keep** `tb_0399_c09` -> `tb_0399_c09`
- **keep** `tb_0399_c10` -> `tb_0399_c10`
- **keep** `tb_0399_c11` -> `tb_0399_c11`

### tb_0400
- **format_consolidated** `tb_0400_c06` -> `tb_0400_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0401
- **keep** `tb_0401_c03` -> `tb_0401_c03`
- **format_consolidated** `tb_0401_c07,tb_0401_c08` -> `tb_0401_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0402
- **format_consolidated** `tb_0402_c10` -> `tb_0402_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0403
- **keep** `tb_0403_c02` -> `tb_0403_c02`
- **keep** `tb_0403_c06` -> `tb_0403_c06`
- **keep** `tb_0403_c10` -> `tb_0403_c09`
- **format_consolidated** `tb_0403_c08` -> `tb_0403_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0404
- **format_consolidated** `tb_0404_c08` -> `tb_0404_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0405
- **split** `tb_0405_c04` -> `tb_0405_c04`
    - was: The response must state that the student incorrectly calculated the value for $x^2$ as 0.045, resulting in an incorrect value for x, x=0.21.
    - now: The response must identify that the student's value for x² (0.045) is incorrect.
- **split** `tb_0405_c04` -> `tb_0405_c05`
    - was: The response must state that the student incorrectly calculated the value for $x^2$ as 0.045, resulting in an incorrect value for x, x=0.21.
    - now: The response must note that this led to an incorrect value of x (x = 0.21).
- **format_consolidated** `tb_0405_c10` -> `tb_0405_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0406
- **format_consolidated** `tb_0406_c13,tb_0406_c14` -> `tb_0406_c14`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0407
- **keep** `tb_0407_c11` -> `tb_0407_c11`
- **keep** `tb_0407_c20` -> `tb_0407_c20`
- **format_consolidated** `tb_0407_c25` -> `tb_0407_c26`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0408
- **keep** `tb_0408_c03` -> `tb_0408_c03`
- **keep** `tb_0408_c05` -> `tb_0408_c05`
- **format_consolidated** `tb_0408_c16` -> `tb_0408_c17`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0409
- **keep** `tb_0409_c01` -> `tb_0409_c01`

### tb_0410
- **format_consolidated** `tb_0410_c13,tb_0410_c14` -> `tb_0410_c14`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0411
- **keep** `tb_0411_c08` -> `tb_0411_c06`
- **keep** `tb_0411_c09` -> `tb_0411_c07`
- **format_consolidated** `tb_0411_c05,tb_0411_c07` -> `tb_0411_c08`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0412
- **keep** `tb_0412_c01` -> `tb_0412_c01`

### tb_0413
- **keep** `tb_0413_c01` -> `tb_0413_c01`
- **keep** `tb_0413_c02` -> `tb_0413_c02`
- **keep** `tb_0413_c03` -> `tb_0413_c03`
- **split** `tb_0413_c04` -> `tb_0413_c04`
    - was: The model must point out the student’s error in (a) not checking the absolute value criterion for geometric convergence and (b) not checking that the limit of the terms is zero for...
    - now: The model must point out the student's error in not checking the absolute value criterion for geometric convergence.
- **split** `tb_0413_c04` -> `tb_0413_c05`
    - was: The model must point out the student’s error in (a) not checking the absolute value criterion for geometric convergence and (b) not checking that the limit of the terms is zero for...
    - now: The model must point out the student's error in not checking that the limit of the terms is zero for all p.
- **split** `tb_0413_c04` -> `tb_0413_c06`
    - was: The model must point out the student’s error in (a) not checking the absolute value criterion for geometric convergence and (b) not checking that the limit of the terms is zero for...
    - now: The model must clarify that if $|p/6| \geq 1$, not only does the AST fail, but the nth term test also proves divergence.
- **keep** `tb_0413_c06` -> `tb_0413_c08`
- **format_consolidated** `tb_0413_c07` -> `tb_0413_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0414
- **split** `tb_0414_c03` -> `tb_0414_c03`
    - was: The model must explain that because the student is using a gas constant R = 62.36 L⋅mmHg⋅K⁻¹⋅mol⁻¹ they must convert the volume from mL to liters and the temperature from degrees C...
    - now: The model must explain that because the student is using a gas constant R = 62.36 L⋅mmHg⋅K⁻¹⋅mol⁻¹ they must convert the volume from mL to liters.
- **split** `tb_0414_c03` -> `tb_0414_c04`
    - was: The model must explain that because the student is using a gas constant R = 62.36 L⋅mmHg⋅K⁻¹⋅mol⁻¹ they must convert the volume from mL to liters and the temperature from degrees C...
    - now: The model must explain that because the student is using a gas constant R = 62.36 L⋅mmHg⋅K⁻¹⋅mol⁻¹ they must convert the temperature from degrees Celsius to Kelvins.
- **format_consolidated** `tb_0414_c11` -> `tb_0414_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0415
- **keep** `tb_0415_c05` -> `tb_0415_c05`
- **format_consolidated** `tb_0415_c07,tb_0415_c08` -> `tb_0415_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0417
- **keep** `tb_0417_c10` -> `tb_0417_c10`

### tb_0418
- **format_consolidated** `tb_0418_c07` -> `tb_0418_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0419
- **keep** `tb_0419_c02` -> `tb_0419_c02`
- **keep** `tb_0419_c04` -> `tb_0419_c04`
- **keep** `tb_0419_c07` -> `tb_0419_c07`
- **format_consolidated** `tb_0419_c09` -> `tb_0419_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0420
- **keep** `tb_0420_c02` -> `tb_0420_c02`
- **keep** `tb_0420_c09` -> `tb_0420_c09`
- **keep** `tb_0420_c10` -> `tb_0420_c10`

### tb_0421
- **format_consolidated** `tb_0421_c02,tb_0421_c06` -> `tb_0421_c09`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0422
- **keep** `tb_0422_c04` -> `tb_0422_c04`
- **keep** `tb_0422_c10` -> `tb_0422_c10`
- **keep** `tb_0422_c13` -> `tb_0422_c13`
- **format_consolidated** `tb_0422_c17` -> `tb_0422_c17`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0424
- **format_consolidated** `tb_0424_c06` -> `tb_0424_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0425
- **format_consolidated** `tb_0425_c06` -> `tb_0425_c07`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0426
- **format_consolidated** `tb_0426_c10` -> `tb_0426_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0427
- **format_consolidated** `tb_0427_c09,tb_0427_c10` -> `tb_0427_c09`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0428
- **format_consolidated** `tb_0428_c08` -> `tb_0428_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0429
- **format_consolidated** `tb_0429_c06` -> `tb_0429_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0431
- **keep** `tb_0431_c01` -> `tb_0431_c01`
- **keep** `tb_0431_c03` -> `tb_0431_c03`
- **keep** `tb_0431_c05` -> `tb_0431_c05`
- **format_consolidated** `tb_0431_c09` -> `tb_0431_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0432
- **format_consolidated** `tb_0432_c07` -> `tb_0432_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0433
- **keep** `tb_0433_c09` -> `tb_0433_c09`

### tb_0434
- **keep** `tb_0434_c06` -> `tb_0434_c06`
- **format_consolidated** `tb_0434_c09` -> `tb_0434_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0435
- **keep** `tb_0435_c02` -> `tb_0435_c02`
- **keep** `tb_0435_c04` -> `tb_0435_c04`
- **format_consolidated** `tb_0435_c13` -> `tb_0435_c13`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0436
- **format_consolidated** `tb_0436_c11` -> `tb_0436_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0437
- **format_consolidated** `tb_0437_c07` -> `tb_0437_c09`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0438
- **keep** `tb_0438_c09` -> `tb_0438_c09`
- **keep** `tb_0438_c11` -> `tb_0438_c11`

### tb_0439
- **keep** `tb_0439_c03` -> `tb_0439_c03`

### tb_0440
- **keep** `tb_0440_c01` -> `tb_0440_c01`

### tb_0441
- **keep** `tb_0441_c05` -> `tb_0441_c05`

### tb_0442
- **keep** `tb_0442_c04` -> `tb_0442_c04`

### tb_0443
- **format_consolidated** `tb_0443_c09` -> `tb_0443_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0444
- **soften** `tb_0444_c01` -> `tb_0444_c01`
    - was: The response must correctly identify that a t-test (Welch’s t-test) is required due to unknown population variances and different sample standard deviations. Using a z-test without...
    - now: The response must correctly identify that a t-test (Welch's t-test) is required due to unknown population variances and different sample standard deviations; using a z-test without justification does not satisfy this criterion.
- **keep** `tb_0444_c02` -> `tb_0444_c02`
- **keep** `tb_0444_c04` -> `tb_0444_c04`
- **format_consolidated** `tb_0444_c06` -> `tb_0444_c08`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0445
- **keep** `tb_0445_c01` -> `tb_0445_c01`
- **keep** `tb_0445_c09` -> `tb_0445_c09`
- **keep** `tb_0445_c10` -> `tb_0445_c10`

### tb_0446
- **format_consolidated** `tb_0446_c06` -> `tb_0446_c06`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0448
- **keep** `tb_0448_c03` -> `tb_0448_c03`
- **keep** `tb_0448_c05` -> `tb_0448_c05`
- **format_consolidated** `tb_0448_c10` -> `tb_0448_c10`
    - was: 1 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0449
- **format_consolidated** `tb_0449_c13` -> `tb_0449_c18`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0451
- **keep** `tb_0451_c01` -> `tb_0451_c01`
- **keep** `tb_0451_c02` -> `tb_0451_c02`
- **keep** `tb_0451_c06` -> `tb_0451_c06`

### tb_0452
- **format_consolidated** `tb_0452_c06,tb_0452_c07` -> `tb_0452_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0453
- **keep** `tb_0453_c01` -> `tb_0453_c01`
- **keep** `tb_0453_c04` -> `tb_0453_c04`
- **keep** `tb_0453_c06` -> `tb_0453_c06`
- **keep** `tb_0453_c07` -> `tb_0453_c07`
- **keep** `tb_0453_c09` -> `tb_0453_c09`
- **keep** `tb_0453_c11` -> `tb_0453_c11`
- **keep** `tb_0453_c15` -> `tb_0453_c15`
- **format_consolidated** `tb_0453_c18` -> `tb_0453_c19`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0454
- **keep** `tb_0454_c06` -> `tb_0454_c06`
- **keep** `tb_0454_c08` -> `tb_0454_c08`
- **keep** `tb_0454_c09` -> `tb_0454_c09`
- **keep** `tb_0454_c11` -> `tb_0454_c11`
- **keep** `tb_0454_c12` -> `tb_0454_c12`
- **keep** `tb_0454_c14` -> `tb_0454_c14`

### tb_0455
- **keep** `tb_0455_c02` -> `tb_0455_c02`
- **keep** `tb_0455_c03` -> `tb_0455_c03`
- **format_consolidated** `tb_0455_c07` -> `tb_0455_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0456
- **keep** `tb_0456_c01` -> `tb_0456_c01`
- **split** `tb_0456_c04` -> `tb_0456_c04`
    - was: The response must identify that the student used an incorrect value of n =1 (instead of 2) and an incorrect reaction quotient expression due to the incorrect balanced equation and ...
    - now: The response must identify that the student used an incorrect value of n = 1 (instead of 2).
- **split** `tb_0456_c04` -> `tb_0456_c05`
    - was: The response must identify that the student used an incorrect value of n =1 (instead of 2) and an incorrect reaction quotient expression due to the incorrect balanced equation and ...
    - now: The response must identify that the student used an incorrect reaction quotient expression due to the incorrect balanced equation, and explain that Q should be [Fe²⁺]²[Zn²⁺]/[Fe³⁺]² instead of [Fe²⁺][Zn²⁺]/[Fe³⁺].
- **format_consolidated** `tb_0456_c07` -> `tb_0456_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0460
- **keep** `tb_0460_c08` -> `tb_0460_c08`
- **keep** `tb_0460_c09` -> `tb_0460_c09`
- **format_consolidated** `tb_0460_c12` -> `tb_0460_c13`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0461
- **keep** `tb_0461_c04` -> `tb_0461_c04`
- **keep** `tb_0461_c09` -> `tb_0461_c07`
- **format_consolidated** `tb_0461_c06,tb_0461_c07` -> `tb_0461_c08`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0462
- **split** `tb_0462_c04` -> `tb_0462_c04`
    - was: The response must describe both ATP roles in the cycle: (i) phophorylation of 3-PGA, and (ii) ATP-driven regenration of RuBP.
    - now: The response must describe the ATP role of phosphorylation of 3-PGA in the Calvin cycle.
- **split** `tb_0462_c04` -> `tb_0462_c05`
    - was: The response must describe both ATP roles in the cycle: (i) phophorylation of 3-PGA, and (ii) ATP-driven regenration of RuBP.
    - now: The response must describe the ATP-driven regeneration of RuBP in the Calvin cycle.
- **format_consolidated** `tb_0462_c13` -> `tb_0462_c14`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0463
- **format_consolidated** `tb_0463_c07` -> `tb_0463_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0464
- **split** `tb_0464_c13` -> `tb_0464_c13`
    - was: The response must state that for (a) increasing temperature, the concentration of SO₂ will increase; for (b) decreasing pressure, the concentration of SO₂ will increase; and for (c...
    - now: The response must state that for (a) increasing temperature, the concentration of SO₂ will increase.
- **split** `tb_0464_c13` -> `tb_0464_c14`
    - was: The response must state that for (a) increasing temperature, the concentration of SO₂ will increase; for (b) decreasing pressure, the concentration of SO₂ will increase; and for (c...
    - now: The response must state that for (b) decreasing pressure, the concentration of SO₂ will increase.
- **split** `tb_0464_c13` -> `tb_0464_c15`
    - was: The response must state that for (a) increasing temperature, the concentration of SO₂ will increase; for (b) decreasing pressure, the concentration of SO₂ will increase; and for (c...
    - now: The response must state that for (c) adding a catalyst, the concentration of SO₂ will remain unchanged.

### tb_0466
- **keep** `tb_0466_c01` -> `tb_0466_c01`
- **keep** `tb_0466_c03` -> `tb_0466_c03`
- **format_consolidated** `tb_0466_c10` -> `tb_0466_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0467
- **keep** `tb_0467_c01` -> `tb_0467_c01`
- **keep** `tb_0467_c02` -> `tb_0467_c02`
- **keep** `tb_0467_c03` -> `tb_0467_c03`
- **keep** `tb_0467_c04` -> `tb_0467_c04`

### tb_0468
- **keep** `tb_0468_c03` -> `tb_0468_c03`

### tb_0469
- **keep** `tb_0469_c05` -> `tb_0469_c05`
- **format_consolidated** `tb_0469_c15` -> `tb_0469_c15`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0471
- **keep** `tb_0471_c06` -> `tb_0471_c05`
- **keep** `tb_0471_c07` -> `tb_0471_c06`
- **keep** `tb_0471_c08` -> `tb_0471_c07`
- **keep** `tb_0471_c15` -> `tb_0471_c14`
- **format_consolidated** `tb_0471_c01` -> `tb_0471_c18`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0472
- **format_consolidated** `tb_0472_c15` -> `tb_0472_c17`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0473
- **keep** `tb_0473_c02` -> `tb_0473_c02`

### tb_0474
- **keep** `tb_0474_c01` -> `tb_0474_c01`

### tb_0475
- **keep** `tb_0475_c02` -> `tb_0475_c02`
- **keep** `tb_0475_c07` -> `tb_0475_c07`
- **keep** `tb_0475_c13` -> `tb_0475_c13`

### tb_0476
- **keep** `tb_0476_c04` -> `tb_0476_c04`
- **format_consolidated** `tb_0476_c08,tb_0476_c09` -> `tb_0476_c08`
    - was: 2 orphan(s) -> aspects=['persona', 'structure', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0477
- **format_consolidated** `tb_0477_c08` -> `tb_0477_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0479
- **keep** `tb_0479_c01` -> `tb_0479_c01`
- **format_consolidated** `tb_0479_c06` -> `tb_0479_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0480
- **keep** `tb_0480_c05` -> `tb_0480_c05`
- **format_consolidated** `tb_0480_c17,tb_0480_c18` -> `tb_0480_c19`
    - was: 2 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0481
- **keep** `tb_0481_c01` -> `tb_0481_c01`
- **keep** `tb_0481_c03` -> `tb_0481_c03`
- **keep** `tb_0481_c04` -> `tb_0481_c04`
- **format_consolidated** `tb_0481_c07` -> `tb_0481_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0482
- **keep** `tb_0482_c04` -> `tb_0482_c04`
- **format_consolidated** `tb_0482_c07,tb_0482_c08` -> `tb_0482_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0483
- **keep** `tb_0483_c03` -> `tb_0483_c03`
- **keep** `tb_0483_c09` -> `tb_0483_c09`
- **keep** `tb_0483_c10` -> `tb_0483_c10`

### tb_0484
- **keep** `tb_0484_c02` -> `tb_0484_c02`

### tb_0485
- **keep** `tb_0485_c03` -> `tb_0485_c03`
- **keep** `tb_0485_c04` -> `tb_0485_c04`

### tb_0488
- **keep** `tb_0488_c01` -> `tb_0488_c01`
- **keep** `tb_0488_c05` -> `tb_0488_c05`
- **format_consolidated** `tb_0488_c09` -> `tb_0488_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0489
- **format_consolidated** `tb_0489_c08` -> `tb_0489_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0490
- **keep** `tb_0490_c07` -> `tb_0490_c07`
- **format_consolidated** `tb_0490_c11` -> `tb_0490_c11`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0491
- **keep** `tb_0491_c03` -> `tb_0491_c03`
- **keep** `tb_0491_c06` -> `tb_0491_c06`
- **split** `tb_0491_c09` -> `tb_0491_c09`
    - was: The response must calculate that p = 1 - 0.707 = 0.293 for the R allele frequency in migrants. 10. The response must provide the correct allele contributions from migrants: 117.2 R...
    - now: The response must calculate that p = 1 - 0.707 = 0.293 for the R allele frequency in migrants.
- **split** `tb_0491_c09` -> `tb_0491_c10`
    - was: The response must calculate that p = 1 - 0.707 = 0.293 for the R allele frequency in migrants. 10. The response must provide the correct allele contributions from migrants: 117.2 R...
    - now: The response must provide the correct allele contributions from migrants: 117.2 R alleles and 282.8 r alleles from 200 plants.
- **format_consolidated** `tb_0491_c15` -> `tb_0491_c16`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0492
- **keep** `tb_0492_c03` -> `tb_0492_c03`
- **format_consolidated** `tb_0492_c05,tb_0492_c06` -> `tb_0492_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0493
- **keep** `tb_0493_c06` -> `tb_0493_c06`
- **keep** `tb_0493_c07` -> `tb_0493_c07`
- **format_consolidated** `tb_0493_c09` -> `tb_0493_c10`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0494
- **keep** `tb_0494_c02` -> `tb_0494_c02`

### tb_0495
- **keep** `tb_0495_c01` -> `tb_0495_c01`
- **keep** `tb_0495_c07` -> `tb_0495_c07`
- **keep** `tb_0495_c09` -> `tb_0495_c09`
- **format_consolidated** `tb_0495_c10` -> `tb_0495_c11`
    - was: 1 orphan(s) -> aspects=['persona', 'code']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0496
- **keep** `tb_0496_c01` -> `tb_0496_c01`
- **keep** `tb_0496_c03` -> `tb_0496_c03`
- **keep** `tb_0496_c04` -> `tb_0496_c04`
- **keep** `tb_0496_c05` -> `tb_0496_c05`
- **format_consolidated** `tb_0496_c06` -> `tb_0496_c07`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0497
- **keep** `tb_0497_c06` -> `tb_0497_c05`
- **format_consolidated** `tb_0497_c05` -> `tb_0497_c06`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0499
- **keep** `tb_0499_c02` -> `tb_0499_c02`
- **keep** `tb_0499_c07` -> `tb_0499_c07`
- **soften** `tb_0499_c11` -> `tb_0499_c11`
    - was: The model response should be formatted using bullet points, headers, tables, graphs and mathematical equations whenever necessary. If any one of these criteria is missing when need...
    - now: The model response should be formatted using bullet points, headers, tables, graphs, and mathematical equations where they aid clarity.
- **format_consolidated** `tb_0499_c11` -> `tb_0499_c11`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0500
- **format_consolidated** `tb_0500_c10` -> `tb_0500_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0501
- **keep** `tb_0501_c06` -> `tb_0501_c06`
- **format_consolidated** `tb_0501_c12` -> `tb_0501_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0502
- **format_consolidated** `tb_0502_c07` -> `tb_0502_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0504
- **format_consolidated** `tb_0504_c12` -> `tb_0504_c17`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0505
- **format_consolidated** `tb_0505_c05,tb_0505_c06` -> `tb_0505_c05`
    - was: 2 orphan(s) -> aspects=['persona', 'math']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0506
- **format_consolidated** `tb_0506_c08` -> `tb_0506_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0507
- **keep** `tb_0507_c04` -> `tb_0507_c04`
- **format_consolidated** `tb_0507_c06` -> `tb_0507_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0508
- **format_consolidated** `tb_0508_c08` -> `tb_0508_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0509
- **keep** `tb_0509_c04` -> `tb_0509_c04`
- **format_consolidated** `tb_0509_c09,tb_0509_c10` -> `tb_0509_c09`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0510
- **keep** `tb_0510_c02` -> `tb_0510_c02`
- **keep** `tb_0510_c09` -> `tb_0510_c08`
- **format_consolidated** `tb_0510_c08` -> `tb_0510_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0511
- **keep** `tb_0511_c09` -> `tb_0511_c09`
- **format_consolidated** `tb_0511_c11` -> `tb_0511_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0513
- **keep** `tb_0513_c01` -> `tb_0513_c01`
- **keep** `tb_0513_c05` -> `tb_0513_c05`

### tb_0514
- **split** `tb_0514_c03` -> `tb_0514_c03`
    - was: The response must identify that the student is unable to proceed because they have not correctly balanced the chemical equation nor understood that they must use moles instead of g...
    - now: The response must identify that the student is unable to proceed because they have not correctly balanced the chemical equation.
- **split** `tb_0514_c03` -> `tb_0514_c04`
    - was: The response must identify that the student is unable to proceed because they have not correctly balanced the chemical equation nor understood that they must use moles instead of g...
    - now: The response must identify that the student is unable to proceed because they have not understood that they must use moles instead of grams for the next steps.

### tb_0515
- **keep** `tb_0515_c07` -> `tb_0515_c07`

### tb_0518
- **keep** `tb_0518_c05` -> `tb_0518_c05`
- **keep** `tb_0518_c10` -> `tb_0518_c10`

### tb_0520
- **format_consolidated** `tb_0520_c07` -> `tb_0520_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0521
- **keep** `tb_0521_c03` -> `tb_0521_c03`
- **keep** `tb_0521_c07` -> `tb_0521_c07`
- **format_consolidated** `tb_0521_c10` -> `tb_0521_c12`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0522
- **keep** `tb_0522_c04` -> `tb_0522_c04`
- **keep** `tb_0522_c05` -> `tb_0522_c05`
- **keep** `tb_0522_c07` -> `tb_0522_c07`

### tb_0524
- **keep** `tb_0524_c03` -> `tb_0524_c03`

### tb_0526
- **keep** `tb_0526_c08` -> `tb_0526_c07`
- **format_consolidated** `tb_0526_c07` -> `tb_0526_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0527
- **keep** `tb_0527_c02` -> `tb_0527_c02`
- **format_consolidated** `tb_0527_c09` -> `tb_0527_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0528
- **keep** `tb_0528_c07` -> `tb_0528_c07`
- **format_consolidated** `tb_0528_c12` -> `tb_0528_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0529
- **keep** `tb_0529_c03` -> `tb_0529_c03`
- **keep** `tb_0529_c06` -> `tb_0529_c06`

### tb_0530
- **keep** `tb_0530_c01` -> `tb_0530_c01`
- **format_consolidated** `tb_0530_c08` -> `tb_0530_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0531
- **format_consolidated** `tb_0531_c16` -> `tb_0531_c16`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0532
- **format_consolidated** `tb_0532_c09` -> `tb_0532_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0533
- **keep** `tb_0533_c06` -> `tb_0533_c05`
- **keep** `tb_0533_c08` -> `tb_0533_c07`
- **format_consolidated** `tb_0533_c03` -> `tb_0533_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0534
- **keep** `tb_0534_c04` -> `tb_0534_c04`
- **format_consolidated** `tb_0534_c06` -> `tb_0534_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0535
- **split** `tb_0535_c04` -> `tb_0535_c04`
    - was: The response must not provide a direct instruction to calculate P(R) or state the formula for the Law of Total Probability (e.g., "P(R) = P(R|A)P(A) + ..."). It must use a guiding ...
    - now: The response must not directly instruct the student to compute P(R) or state the Law of Total Probability formula.
- **split** `tb_0535_c04` -> `tb_0535_c05`
    - was: The response must not provide a direct instruction to calculate P(R) or state the formula for the Law of Total Probability (e.g., "P(R) = P(R|A)P(A) + ..."). It must use a guiding ...
    - now: The response must instead prompt the student with a guiding question that moves them toward the approach.

### tb_0537
- **keep** `tb_0537_c03` -> `tb_0537_c03`
- **format_consolidated** `tb_0537_c06,tb_0537_c07` -> `tb_0537_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0538
- **format_consolidated** `tb_0538_c05,tb_0538_c07` -> `tb_0538_c06`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0541
- **format_consolidated** `tb_0541_c08,tb_0541_c09` -> `tb_0541_c09`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0542
- **split** `tb_0542_c02` -> `tb_0542_c02`
    - was: The response must acknowledge that the student arrived at the correct formula and directly address their confusion about why the flawed reasoning is still incorrect (e.g., by expla...
    - now: The response must acknowledge that the student arrived at the correct formula.
- **split** `tb_0542_c02` -> `tb_0542_c03`
    - was: The response must acknowledge that the student arrived at the correct formula and directly address their confusion about why the flawed reasoning is still incorrect (e.g., by expla...
    - now: The response must directly address the student's confusion about why the flawed reasoning is still incorrect (e.g., by explaining that a correct formula can sometimes result from an incorrect method).
- **keep** `tb_0542_c04` -> `tb_0542_c05`
- **format_consolidated** `tb_0542_c07` -> `tb_0542_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0543
- **keep** `tb_0543_c03` -> `tb_0543_c03`

### tb_0544
- **keep** `tb_0544_c02` -> `tb_0544_c02`
- **keep** `tb_0544_c07` -> `tb_0544_c07`

### tb_0545
- **keep** `tb_0545_c01` -> `tb_0545_c01`
- **keep** `tb_0545_c02` -> `tb_0545_c02`
- **keep** `tb_0545_c03` -> `tb_0545_c03`
- **format_consolidated** `tb_0545_c07` -> `tb_0545_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0546
- **keep** `tb_0546_c02` -> `tb_0546_c02`
- **keep** `tb_0546_c03` -> `tb_0546_c03`

### tb_0548
- **keep** `tb_0548_c02` -> `tb_0548_c02`
- **keep** `tb_0548_c03` -> `tb_0548_c03`

### tb_0549
- **split** `tb_0549_c02` -> `tb_0549_c02`
    - was: The response must acknowledge the student's correct progress, specifically mentioning their calculation of P1's velocity and their correct setup of the absolute value integral for ...
    - now: The response must acknowledge the student's correct calculation of P1's velocity.
- **split** `tb_0549_c02` -> `tb_0549_c03`
    - was: The response must acknowledge the student's correct progress, specifically mentioning their calculation of P1's velocity and their correct setup of the absolute value integral for ...
    - now: The response must acknowledge the student's correct setup of the absolute value integral for P2's distance.

### tb_0551
- **format_consolidated** `tb_0551_c07` -> `tb_0551_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0552
- **split** `tb_0552_c02` -> `tb_0552_c02`
    - was: The response must state that the student mis-read the question by (a) treating both children as blue-eyed and (b) ignoring the ordered wording "first ... then ...".
    - now: The response must state that the student mis-read the question by treating both children as blue-eyed.
- **split** `tb_0552_c02` -> `tb_0552_c03`
    - was: The response must state that the student mis-read the question by (a) treating both children as blue-eyed and (b) ignoring the ordered wording "first ... then ...".
    - now: The response must state that the student mis-read the question by ignoring the ordered wording "first ... then ...".
- **keep** `tb_0552_c03` -> `tb_0552_c04`
- **keep** `tb_0552_c04` -> `tb_0552_c05`
- **keep** `tb_0552_c05` -> `tb_0552_c06`

### tb_0553
- **keep** `tb_0553_c01` -> `tb_0553_c01`
- **format_consolidated** `tb_0553_c08` -> `tb_0553_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0554
- **format_consolidated** `tb_0554_c09` -> `tb_0554_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0555
- **split** `tb_0555_c03` -> `tb_0555_c03`
    - was: The response must provide a hint by explaining that since the equation they are trying to solve (\sec^2\left(\frac{t}{2}\right) - e^{-t^2} + 3 = \frac{5t^2}{2} + \sin(t)) involves ...
    - now: The response must point out that the equation mixes trigonometric, exponential, and polynomial terms.
- **split** `tb_0555_c03` -> `tb_0555_c04`
    - was: The response must provide a hint by explaining that since the equation they are trying to solve (\sec^2\left(\frac{t}{2}\right) - e^{-t^2} + 3 = \frac{5t^2}{2} + \sin(t)) involves ...
    - now: The response must convey that such an equation is unlikely to have a closed-form solution.
- **split** `tb_0555_c03` -> `tb_0555_c05`
    - was: The response must provide a hint by explaining that since the equation they are trying to solve (\sec^2\left(\frac{t}{2}\right) - e^{-t^2} + 3 = \frac{5t^2}{2} + \sin(t)) involves ...
    - now: The response must suggest an alternative approach, such as graphing or a numerical method.
- **keep** `tb_0555_c05` -> `tb_0555_c07`
- **format_consolidated** `tb_0555_c13,tb_0555_c14` -> `tb_0555_c16`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0556
- **format_consolidated** `tb_0556_c09` -> `tb_0556_c10`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0557
- **keep** `tb_0557_c08` -> `tb_0557_c08`

### tb_0558
- **keep** `tb_0558_c08` -> `tb_0558_c08`

### tb_0559
- **keep** `tb_0559_c09` -> `tb_0559_c09`

### tb_0560
- **keep** `tb_0560_c09` -> `tb_0560_c09`

### tb_0564
- **format_consolidated** `tb_0564_c13` -> `tb_0564_c15`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0565
- **split** `tb_0565_c04` -> `tb_0565_c04`
    - was: The model must acknowledge that the student is correctly using related rates by expressing surface area and volume in terms of radius, correctly setup the derivative of the volume ...
    - now: The model must acknowledge that the student is correctly using related rates by expressing surface area and volume in terms of radius.
- **split** `tb_0565_c04` -> `tb_0565_c05`
    - was: The model must acknowledge that the student is correctly using related rates by expressing surface area and volume in terms of radius, correctly setup the derivative of the volume ...
    - now: The model must acknowledge that the student correctly set up the derivative of the volume formula to find the volume's rate of change.
- **split** `tb_0565_c04` -> `tb_0565_c06`
    - was: The model must acknowledge that the student is correctly using related rates by expressing surface area and volume in terms of radius, correctly setup the derivative of the volume ...
    - now: The model must acknowledge that the student determined that the chain rule must be applied.
- **keep** `tb_0565_c06` -> `tb_0565_c08`
- **keep** `tb_0565_c08` -> `tb_0565_c10`
- **format_consolidated** `tb_0565_c11` -> `tb_0565_c13`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0566
- **keep** `tb_0566_c03` -> `tb_0566_c03`
- **format_consolidated** `tb_0566_c08` -> `tb_0566_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0568
- **format_consolidated** `tb_0568_c12` -> `tb_0568_c12`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0569
- **format_consolidated** `tb_0569_c10` -> `tb_0569_c10`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0570
- **keep** `tb_0570_c03` -> `tb_0570_c03`
- **keep** `tb_0570_c04` -> `tb_0570_c04`
- **format_consolidated** `tb_0570_c07` -> `tb_0570_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0573
- **keep** `tb_0573_c06` -> `tb_0573_c06`
- **format_consolidated** `tb_0573_c16` -> `tb_0573_c16`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0574
- **keep** `tb_0574_c01` -> `tb_0574_c01`

### tb_0575
- **keep** `tb_0575_c02` -> `tb_0575_c02`
- **format_consolidated** `tb_0575_c04` -> `tb_0575_c05`
    - was: 1 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0576
- **format_consolidated** `tb_0576_c13` -> `tb_0576_c14`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0578
- **keep** `tb_0578_c02` -> `tb_0578_c02`

### tb_0579
- **format_consolidated** `tb_0579_c08` -> `tb_0579_c08`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0580
- **format_consolidated** `tb_0580_c07` -> `tb_0580_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0581
- **format_consolidated** `tb_0581_c02,tb_0581_c04` -> `tb_0581_c12`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0582
- **keep** `tb_0582_c05` -> `tb_0582_c05`

### tb_0583
- **keep** `tb_0583_c02` -> `tb_0583_c02`

### tb_0584
- **keep** `tb_0584_c04` -> `tb_0584_c04`
- **keep** `tb_0584_c08` -> `tb_0584_c08`

### tb_0585
- **format_consolidated** `tb_0585_c07,tb_0585_c08` -> `tb_0585_c09`
    - was: 2 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0586
- **format_consolidated** `tb_0586_c01,tb_0586_c06` -> `tb_0586_c07`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0589
- **format_consolidated** `tb_0589_c07` -> `tb_0589_c07`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0591
- **keep** `tb_0591_c02` -> `tb_0591_c02`
- **keep** `tb_0591_c07` -> `tb_0591_c07`
- **format_consolidated** `tb_0591_c13` -> `tb_0591_c14`
    - was: 1 orphan(s) -> aspects=['generic']
    - now: The response should follow tutoring presentation conventions: be clearly and readably presented. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0592
- **format_consolidated** `tb_0592_c13` -> `tb_0592_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0593
- **keep** `tb_0593_c02` -> `tb_0593_c02`
- **keep** `tb_0593_c06` -> `tb_0593_c06`
- **keep** `tb_0593_c08` -> `tb_0593_c07`
- **keep** `tb_0593_c11` -> `tb_0593_c10`
- **format_consolidated** `tb_0593_c07` -> `tb_0593_c12`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0594
- **keep** `tb_0594_c03` -> `tb_0594_c03`

### tb_0596
- **keep** `tb_0596_c06` -> `tb_0596_c06`

### tb_0600
- **split** `tb_0600_c02` -> `tb_0600_c02`
    - was: The response must identify why the student is stuck, namely a misunderstanding of two key ideas: (a) that all species’ concentrations are needed for Kc, and (b) that stoichiometric...
    - now: The response must identify that the student is stuck due to a misunderstanding that all species' concentrations are needed for Kc.
- **split** `tb_0600_c02` -> `tb_0600_c03`
    - was: The response must identify why the student is stuck, namely a misunderstanding of two key ideas: (a) that all species’ concentrations are needed for Kc, and (b) that stoichiometric...
    - now: The response must identify that the student is stuck due to a misunderstanding that stoichiometric coefficients translate to exponents, not multipliers.
- **keep** `tb_0600_c06` -> `tb_0600_c07`

### tb_0601
- **format_consolidated** `tb_0601_c10` -> `tb_0601_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0603
- **format_consolidated** `tb_0603_c07` -> `tb_0603_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0605
- **keep** `tb_0605_c03` -> `tb_0605_c03`
- **format_consolidated** `tb_0605_c09` -> `tb_0605_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0607
- **keep** `tb_0607_c04` -> `tb_0607_c04`

### tb_0609
- **split** `tb_0609_c01` -> `tb_0609_c01`
    - was: The response must acknowledge the student correct identified the purpose of the reagents and the correct intermediate molecule, 3-nitrotoluene.
    - now: The response must acknowledge the student correctly identified the purpose of the reagents.
- **split** `tb_0609_c01` -> `tb_0609_c02`
    - was: The response must acknowledge the student correct identified the purpose of the reagents and the correct intermediate molecule, 3-nitrotoluene.
    - now: The response must acknowledge the student correctly identified the correct intermediate molecule, 3-nitrotoluene.

### tb_0610
- **keep** `tb_0610_c03` -> `tb_0610_c03`

### tb_0612
- **format_consolidated** `tb_0612_c06` -> `tb_0612_c07`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0615
- **split** `tb_0615_c03` -> `tb_0615_c03`
    - was: The response must provide a hint for Part (a) that guides the student to consider the Integration by Parts method and formula ($\int u \text{du} = uv - \int v \text{du}$) alongside...
    - now: The response must hint that Part (a) can be approached with integration by parts (∫u dv = uv − ∫v du).
- **split** `tb_0615_c03` -> `tb_0615_c04`
    - was: The response must provide a hint for Part (a) that guides the student to consider the Integration by Parts method and formula ($\int u \text{du} = uv - \int v \text{du}$) alongside...
    - now: The response must also point the student toward u-substitution as a complementary technique.
- **keep** `tb_0615_c04` -> `tb_0615_c05`
- **format_consolidated** `tb_0615_c08` -> `tb_0615_c12`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0616
- **keep** `tb_0616_c01` -> `tb_0616_c01`
- **split** `tb_0616_c02` -> `tb_0616_c02`
    - was: The response must acknowledge the correct ideas that (a) continuous light triggers stomatal opening and (b) low internal CO2 usually keeps stomata open.
    - now: The response must acknowledge the correct idea that continuous light triggers stomatal opening.
- **split** `tb_0616_c02` -> `tb_0616_c03`
    - was: The response must acknowledge the correct ideas that (a) continuous light triggers stomatal opening and (b) low internal CO2 usually keeps stomata open.
    - now: The response must acknowledge the correct idea that low internal CO2 usually keeps stomata open.
- **keep** `tb_0616_c04` -> `tb_0616_c05`
- **format_consolidated** `tb_0616_c08` -> `tb_0616_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0617
- **keep** `tb_0617_c02` -> `tb_0617_c02`
- **keep** `tb_0617_c03` -> `tb_0617_c03`
- **keep** `tb_0617_c06` -> `tb_0617_c06`
- **keep** `tb_0617_c07` -> `tb_0617_c07`
- **format_consolidated** `tb_0617_c08` -> `tb_0617_c11`
    - was: 1 orphan(s) -> aspects=['persona']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice). (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0618
- **keep** `tb_0618_c01` -> `tb_0618_c01`
- **keep** `tb_0618_c07` -> `tb_0618_c07`
- **keep** `tb_0618_c09` -> `tb_0618_c09`
- **format_consolidated** `tb_0618_c13` -> `tb_0618_c13`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0619
- **keep** `tb_0619_c08` -> `tb_0619_c08`

### tb_0620
- **keep** `tb_0620_c06` -> `tb_0620_c06`
- **format_consolidated** `tb_0620_c11` -> `tb_0620_c11`
    - was: 1 orphan(s) -> aspects=['math', 'code']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0622
- **format_consolidated** `tb_0622_c12` -> `tb_0622_c13`
    - was: 1 orphan(s) -> aspects=['generic']
    - now: The response should follow tutoring presentation conventions: be clearly and readably presented. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0623
- **keep** `tb_0623_c01` -> `tb_0623_c01`
- **keep** `tb_0623_c03` -> `tb_0623_c03`

### tb_0625
- **format_consolidated** `tb_0625_c09` -> `tb_0625_c11`
    - was: 1 orphan(s) -> aspects=['structure', 'math']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0626
- **keep** `tb_0626_c11` -> `tb_0626_c11`
- **keep** `tb_0626_c15` -> `tb_0626_c15`

### tb_0627
- **keep** `tb_0627_c03` -> `tb_0627_c03`

### tb_0628
- **keep** `tb_0628_c07` -> `tb_0628_c07`

### tb_0629
- **format_consolidated** `tb_0629_c08` -> `tb_0629_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0630
- **keep** `tb_0630_c02` -> `tb_0630_c02`
- **keep** `tb_0630_c07` -> `tb_0630_c07`
- **format_consolidated** `tb_0630_c10` -> `tb_0630_c11`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0631
- **keep** `tb_0631_c04` -> `tb_0631_c04`

### tb_0632
- **format_consolidated** `tb_0632_c03` -> `tb_0632_c13`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0634
- **keep** `tb_0634_c01` -> `tb_0634_c01`

### tb_0635
- **format_consolidated** `tb_0635_c06` -> `tb_0635_c06`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0636
- **keep** `tb_0636_c04` -> `tb_0636_c04`

### tb_0637
- **keep** `tb_0637_c03` -> `tb_0637_c03`

### tb_0638
- **keep** `tb_0638_c03` -> `tb_0638_c03`

### tb_0640
- **format_consolidated** `tb_0640_c09` -> `tb_0640_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0641
- **keep** `tb_0641_c09` -> `tb_0641_c08`
- **format_consolidated** `tb_0641_c07` -> `tb_0641_c09`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0642
- **keep** `tb_0642_c06` -> `tb_0642_c06`

### tb_0643
- **keep** `tb_0643_c03` -> `tb_0643_c03`

### tb_0645
- **keep** `tb_0645_c01` -> `tb_0645_c01`
- **keep** `tb_0645_c03` -> `tb_0645_c03`

### tb_0646
- **keep** `tb_0646_c08` -> `tb_0646_c08`

### tb_0647
- **keep** `tb_0647_c04` -> `tb_0647_c04`
- **format_consolidated** `tb_0647_c14` -> `tb_0647_c15`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0648
- **keep** `tb_0648_c05` -> `tb_0648_c05`
- **keep** `tb_0648_c09` -> `tb_0648_c09`

### tb_0649
- **format_consolidated** `tb_0649_c05` -> `tb_0649_c05`
    - was: 1 orphan(s) -> aspects=['math']
    - now: The response should follow tutoring presentation conventions: render mathematical expressions in LaTeX. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0650
- **keep** `tb_0650_c01` -> `tb_0650_c01`
- **keep** `tb_0650_c05` -> `tb_0650_c05`
- **keep** `tb_0650_c06` -> `tb_0650_c06`

### tb_0651
- **keep** `tb_0651_c04` -> `tb_0651_c04`

### tb_0652
- **keep** `tb_0652_c02` -> `tb_0652_c02`
- **keep** `tb_0652_c03` -> `tb_0652_c03`

### tb_0654
- **format_consolidated** `tb_0654_c06` -> `tb_0654_c06`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0655
- **format_consolidated** `tb_0655_c08` -> `tb_0655_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0656
- **keep** `tb_0656_c02` -> `tb_0656_c02`

### tb_0657
- **format_consolidated** `tb_0657_c08,tb_0657_c09` -> `tb_0657_c08`
    - was: 2 orphan(s) -> aspects=['persona', 'structure']
    - now: The response should follow tutoring presentation conventions: address the student in the second person (conversational tutor voice); use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0658
- **keep** `tb_0658_c06` -> `tb_0658_c06`
- **keep** `tb_0658_c07` -> `tb_0658_c07`

### tb_0659
- **keep** `tb_0659_c05` -> `tb_0659_c05`
- **format_consolidated** `tb_0659_c07` -> `tb_0659_c08`
    - was: 1 orphan(s) -> aspects=['structure']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0662
- **format_consolidated** `tb_0662_c10` -> `tb_0662_c10`
    - was: 1 orphan(s) -> aspects=['structure', 'code']
    - now: The response should follow tutoring presentation conventions: use clear Markdown structure (e.g., headings, bullets, or sections) where it aids clarity; present code in fenced code blocks with correct syntax. (Style/presentation only; not a content, diagnosis, or scaffolding skill.)

### tb_0001
- **presentation_added** `` -> `tb_0001_c11`

### tb_0002
- **presentation_added** `` -> `tb_0002_c22`

### tb_0005
- **presentation_added** `` -> `tb_0005_c09`

### tb_0006
- **presentation_added** `` -> `tb_0006_c10`

### tb_0007
- **presentation_added** `` -> `tb_0007_c10`

### tb_0009
- **presentation_added** `` -> `tb_0009_c09`

### tb_0013
- **presentation_added** `` -> `tb_0013_c12`

### tb_0016
- **presentation_added** `` -> `tb_0016_c08`

### tb_0017
- **presentation_added** `` -> `tb_0017_c22`

### tb_0019
- **presentation_added** `` -> `tb_0019_c14`

### tb_0021
- **presentation_added** `` -> `tb_0021_c11`

### tb_0025
- **presentation_added** `` -> `tb_0025_c13`

### tb_0026
- **presentation_added** `` -> `tb_0026_c17`

### tb_0027
- **presentation_added** `` -> `tb_0027_c13`

### tb_0028
- **presentation_added** `` -> `tb_0028_c13`

### tb_0029
- **presentation_added** `` -> `tb_0029_c08`

### tb_0030
- **presentation_added** `` -> `tb_0030_c09`

### tb_0031
- **presentation_added** `` -> `tb_0031_c14`

### tb_0033
- **presentation_added** `` -> `tb_0033_c12`

### tb_0036
- **presentation_added** `` -> `tb_0036_c11`

### tb_0039
- **presentation_added** `` -> `tb_0039_c10`

### tb_0041
- **presentation_added** `` -> `tb_0041_c09`

### tb_0042
- **presentation_added** `` -> `tb_0042_c11`

### tb_0044
- **presentation_added** `` -> `tb_0044_c08`

### tb_0045
- **presentation_added** `` -> `tb_0045_c14`

### tb_0047
- **presentation_added** `` -> `tb_0047_c12`

### tb_0048
- **presentation_added** `` -> `tb_0048_c12`

### tb_0049
- **presentation_added** `` -> `tb_0049_c07`

### tb_0050
- **presentation_added** `` -> `tb_0050_c10`

### tb_0052
- **presentation_added** `` -> `tb_0052_c12`

### tb_0058
- **presentation_added** `` -> `tb_0058_c08`

### tb_0060
- **presentation_added** `` -> `tb_0060_c04`

### tb_0062
- **presentation_added** `` -> `tb_0062_c08`

### tb_0063
- **presentation_added** `` -> `tb_0063_c14`

### tb_0064
- **presentation_added** `` -> `tb_0064_c11`

### tb_0065
- **presentation_added** `` -> `tb_0065_c09`

### tb_0066
- **presentation_added** `` -> `tb_0066_c08`

### tb_0069
- **presentation_added** `` -> `tb_0069_c18`

### tb_0072
- **presentation_added** `` -> `tb_0072_c11`

### tb_0073
- **presentation_added** `` -> `tb_0073_c08`

### tb_0074
- **presentation_added** `` -> `tb_0074_c11`

### tb_0076
- **presentation_added** `` -> `tb_0076_c06`

### tb_0077
- **presentation_added** `` -> `tb_0077_c09`

### tb_0078
- **presentation_added** `` -> `tb_0078_c19`

### tb_0080
- **presentation_added** `` -> `tb_0080_c13`

### tb_0082
- **presentation_added** `` -> `tb_0082_c13`

### tb_0083
- **presentation_added** `` -> `tb_0083_c09`

### tb_0085
- **presentation_added** `` -> `tb_0085_c10`

### tb_0088
- **presentation_added** `` -> `tb_0088_c10`

### tb_0089
- **presentation_added** `` -> `tb_0089_c09`

### tb_0090
- **presentation_added** `` -> `tb_0090_c08`

### tb_0093
- **presentation_added** `` -> `tb_0093_c13`

### tb_0094
- **presentation_added** `` -> `tb_0094_c06`

### tb_0097
- **presentation_added** `` -> `tb_0097_c07`

### tb_0100
- **presentation_added** `` -> `tb_0100_c09`

### tb_0101
- **presentation_added** `` -> `tb_0101_c08`

### tb_0102
- **presentation_added** `` -> `tb_0102_c19`

### tb_0104
- **presentation_added** `` -> `tb_0104_c19`

### tb_0109
- **presentation_added** `` -> `tb_0109_c14`

### tb_0112
- **presentation_added** `` -> `tb_0112_c08`

### tb_0114
- **presentation_added** `` -> `tb_0114_c08`

### tb_0116
- **presentation_added** `` -> `tb_0116_c08`

### tb_0117
- **presentation_added** `` -> `tb_0117_c13`

### tb_0118
- **presentation_added** `` -> `tb_0118_c09`

### tb_0119
- **presentation_added** `` -> `tb_0119_c08`

### tb_0122
- **presentation_added** `` -> `tb_0122_c13`

### tb_0123
- **presentation_added** `` -> `tb_0123_c04`

### tb_0126
- **presentation_added** `` -> `tb_0126_c11`

### tb_0129
- **presentation_added** `` -> `tb_0129_c16`

### tb_0130
- **presentation_added** `` -> `tb_0130_c10`

### tb_0131
- **presentation_added** `` -> `tb_0131_c12`

### tb_0133
- **presentation_added** `` -> `tb_0133_c12`

### tb_0134
- **presentation_added** `` -> `tb_0134_c07`

### tb_0136
- **presentation_added** `` -> `tb_0136_c11`

### tb_0137
- **presentation_added** `` -> `tb_0137_c07`

### tb_0138
- **presentation_added** `` -> `tb_0138_c08`

### tb_0139
- **presentation_added** `` -> `tb_0139_c10`

### tb_0140
- **presentation_added** `` -> `tb_0140_c08`

### tb_0141
- **presentation_added** `` -> `tb_0141_c12`

### tb_0146
- **presentation_added** `` -> `tb_0146_c11`

### tb_0147
- **presentation_added** `` -> `tb_0147_c14`

### tb_0149
- **presentation_added** `` -> `tb_0149_c13`

### tb_0150
- **presentation_added** `` -> `tb_0150_c10`

### tb_0156
- **presentation_added** `` -> `tb_0156_c10`

### tb_0157
- **presentation_added** `` -> `tb_0157_c11`

### tb_0158
- **presentation_added** `` -> `tb_0158_c05`

### tb_0159
- **presentation_added** `` -> `tb_0159_c11`

### tb_0161
- **presentation_added** `` -> `tb_0161_c08`

### tb_0166
- **presentation_added** `` -> `tb_0166_c15`

### tb_0167
- **presentation_added** `` -> `tb_0167_c06`

### tb_0168
- **presentation_added** `` -> `tb_0168_c11`

### tb_0170
- **presentation_added** `` -> `tb_0170_c10`

### tb_0172
- **presentation_added** `` -> `tb_0172_c10`

### tb_0174
- **presentation_added** `` -> `tb_0174_c11`

### tb_0175
- **presentation_added** `` -> `tb_0175_c07`

### tb_0176
- **presentation_added** `` -> `tb_0176_c11`

### tb_0177
- **presentation_added** `` -> `tb_0177_c20`

### tb_0179
- **presentation_added** `` -> `tb_0179_c16`

### tb_0180
- **presentation_added** `` -> `tb_0180_c08`

### tb_0181
- **presentation_added** `` -> `tb_0181_c12`

### tb_0183
- **presentation_added** `` -> `tb_0183_c09`

### tb_0184
- **presentation_added** `` -> `tb_0184_c08`

### tb_0185
- **presentation_added** `` -> `tb_0185_c09`

### tb_0186
- **presentation_added** `` -> `tb_0186_c15`

### tb_0187
- **presentation_added** `` -> `tb_0187_c10`

### tb_0188
- **presentation_added** `` -> `tb_0188_c09`

### tb_0190
- **presentation_added** `` -> `tb_0190_c15`

### tb_0191
- **presentation_added** `` -> `tb_0191_c12`

### tb_0192
- **presentation_added** `` -> `tb_0192_c15`

### tb_0193
- **presentation_added** `` -> `tb_0193_c11`

### tb_0194
- **presentation_added** `` -> `tb_0194_c11`

### tb_0195
- **presentation_added** `` -> `tb_0195_c08`

### tb_0198
- **presentation_added** `` -> `tb_0198_c10`

### tb_0200
- **presentation_added** `` -> `tb_0200_c11`

### tb_0201
- **presentation_added** `` -> `tb_0201_c17`

### tb_0203
- **presentation_added** `` -> `tb_0203_c06`

### tb_0205
- **presentation_added** `` -> `tb_0205_c08`

### tb_0206
- **presentation_added** `` -> `tb_0206_c08`

### tb_0207
- **presentation_added** `` -> `tb_0207_c10`

### tb_0208
- **presentation_added** `` -> `tb_0208_c10`

### tb_0210
- **presentation_added** `` -> `tb_0210_c09`

### tb_0212
- **presentation_added** `` -> `tb_0212_c11`

### tb_0215
- **presentation_added** `` -> `tb_0215_c12`

### tb_0216
- **presentation_added** `` -> `tb_0216_c12`

### tb_0218
- **presentation_added** `` -> `tb_0218_c10`

### tb_0219
- **presentation_added** `` -> `tb_0219_c12`

### tb_0220
- **presentation_added** `` -> `tb_0220_c06`

### tb_0226
- **presentation_added** `` -> `tb_0226_c20`

### tb_0228
- **presentation_added** `` -> `tb_0228_c14`

### tb_0230
- **presentation_added** `` -> `tb_0230_c13`

### tb_0231
- **presentation_added** `` -> `tb_0231_c12`

### tb_0233
- **presentation_added** `` -> `tb_0233_c10`

### tb_0236
- **presentation_added** `` -> `tb_0236_c11`

### tb_0237
- **presentation_added** `` -> `tb_0237_c09`

### tb_0239
- **presentation_added** `` -> `tb_0239_c09`

### tb_0240
- **presentation_added** `` -> `tb_0240_c12`

### tb_0241
- **presentation_added** `` -> `tb_0241_c18`

### tb_0242
- **presentation_added** `` -> `tb_0242_c07`

### tb_0246
- **presentation_added** `` -> `tb_0246_c06`

### tb_0247
- **presentation_added** `` -> `tb_0247_c08`

### tb_0249
- **presentation_added** `` -> `tb_0249_c06`

### tb_0250
- **presentation_added** `` -> `tb_0250_c08`

### tb_0252
- **presentation_added** `` -> `tb_0252_c07`

### tb_0253
- **presentation_added** `` -> `tb_0253_c11`

### tb_0254
- **presentation_added** `` -> `tb_0254_c10`

### tb_0256
- **presentation_added** `` -> `tb_0256_c12`

### tb_0257
- **presentation_added** `` -> `tb_0257_c13`

### tb_0259
- **presentation_added** `` -> `tb_0259_c17`

### tb_0263
- **presentation_added** `` -> `tb_0263_c13`

### tb_0267
- **presentation_added** `` -> `tb_0267_c20`

### tb_0269
- **presentation_added** `` -> `tb_0269_c12`

### tb_0272
- **presentation_added** `` -> `tb_0272_c08`

### tb_0274
- **presentation_added** `` -> `tb_0274_c17`

### tb_0275
- **presentation_added** `` -> `tb_0275_c10`

### tb_0276
- **presentation_added** `` -> `tb_0276_c14`

### tb_0277
- **presentation_added** `` -> `tb_0277_c09`

### tb_0278
- **presentation_added** `` -> `tb_0278_c16`

### tb_0281
- **presentation_added** `` -> `tb_0281_c09`

### tb_0283
- **presentation_added** `` -> `tb_0283_c11`

### tb_0284
- **presentation_added** `` -> `tb_0284_c07`

### tb_0286
- **presentation_added** `` -> `tb_0286_c13`

### tb_0287
- **presentation_added** `` -> `tb_0287_c08`

### tb_0288
- **presentation_added** `` -> `tb_0288_c07`

### tb_0290
- **presentation_added** `` -> `tb_0290_c10`

### tb_0292
- **presentation_added** `` -> `tb_0292_c08`

### tb_0293
- **presentation_added** `` -> `tb_0293_c10`

### tb_0294
- **presentation_added** `` -> `tb_0294_c10`

### tb_0296
- **presentation_added** `` -> `tb_0296_c11`

### tb_0301
- **presentation_added** `` -> `tb_0301_c12`

### tb_0303
- **presentation_added** `` -> `tb_0303_c08`

### tb_0304
- **presentation_added** `` -> `tb_0304_c10`

### tb_0305
- **presentation_added** `` -> `tb_0305_c09`

### tb_0307
- **presentation_added** `` -> `tb_0307_c16`

### tb_0309
- **presentation_added** `` -> `tb_0309_c10`

### tb_0311
- **presentation_added** `` -> `tb_0311_c08`

### tb_0313
- **presentation_added** `` -> `tb_0313_c08`

### tb_0314
- **presentation_added** `` -> `tb_0314_c13`

### tb_0315
- **presentation_added** `` -> `tb_0315_c08`

### tb_0318
- **presentation_added** `` -> `tb_0318_c13`

### tb_0319
- **presentation_added** `` -> `tb_0319_c11`

### tb_0323
- **presentation_added** `` -> `tb_0323_c06`

### tb_0324
- **presentation_added** `` -> `tb_0324_c07`

### tb_0325
- **presentation_added** `` -> `tb_0325_c10`

### tb_0326
- **presentation_added** `` -> `tb_0326_c06`

### tb_0327
- **presentation_added** `` -> `tb_0327_c09`

### tb_0330
- **presentation_added** `` -> `tb_0330_c25`

### tb_0332
- **presentation_added** `` -> `tb_0332_c09`

### tb_0333
- **presentation_added** `` -> `tb_0333_c11`

### tb_0337
- **presentation_added** `` -> `tb_0337_c09`

### tb_0338
- **presentation_added** `` -> `tb_0338_c09`

### tb_0342
- **presentation_added** `` -> `tb_0342_c09`

### tb_0345
- **presentation_added** `` -> `tb_0345_c13`

### tb_0347
- **presentation_added** `` -> `tb_0347_c11`

### tb_0348
- **presentation_added** `` -> `tb_0348_c09`

### tb_0349
- **presentation_added** `` -> `tb_0349_c09`

### tb_0350
- **presentation_added** `` -> `tb_0350_c08`

### tb_0351
- **presentation_added** `` -> `tb_0351_c10`

### tb_0352
- **presentation_added** `` -> `tb_0352_c08`

### tb_0353
- **presentation_added** `` -> `tb_0353_c06`

### tb_0355
- **presentation_added** `` -> `tb_0355_c10`

### tb_0356
- **presentation_added** `` -> `tb_0356_c08`

### tb_0360
- **presentation_added** `` -> `tb_0360_c12`

### tb_0363
- **presentation_added** `` -> `tb_0363_c16`

### tb_0365
- **presentation_added** `` -> `tb_0365_c09`

### tb_0367
- **presentation_added** `` -> `tb_0367_c12`

### tb_0369
- **presentation_added** `` -> `tb_0369_c22`

### tb_0373
- **presentation_added** `` -> `tb_0373_c13`

### tb_0376
- **presentation_added** `` -> `tb_0376_c19`

### tb_0377
- **presentation_added** `` -> `tb_0377_c08`

### tb_0378
- **presentation_added** `` -> `tb_0378_c15`

### tb_0381
- **presentation_added** `` -> `tb_0381_c15`

### tb_0383
- **presentation_added** `` -> `tb_0383_c10`

### tb_0385
- **presentation_added** `` -> `tb_0385_c13`

### tb_0386
- **presentation_added** `` -> `tb_0386_c10`

### tb_0387
- **presentation_added** `` -> `tb_0387_c08`

### tb_0390
- **presentation_added** `` -> `tb_0390_c10`

### tb_0393
- **presentation_added** `` -> `tb_0393_c12`

### tb_0394
- **presentation_added** `` -> `tb_0394_c13`

### tb_0396
- **presentation_added** `` -> `tb_0396_c13`

### tb_0397
- **presentation_added** `` -> `tb_0397_c08`

### tb_0398
- **presentation_added** `` -> `tb_0398_c09`

### tb_0399
- **presentation_added** `` -> `tb_0399_c14`

### tb_0409
- **presentation_added** `` -> `tb_0409_c07`

### tb_0412
- **presentation_added** `` -> `tb_0412_c08`

### tb_0416
- **presentation_added** `` -> `tb_0416_c05`

### tb_0417
- **presentation_added** `` -> `tb_0417_c12`

### tb_0420
- **presentation_added** `` -> `tb_0420_c12`

### tb_0423
- **presentation_added** `` -> `tb_0423_c11`

### tb_0430
- **presentation_added** `` -> `tb_0430_c10`

### tb_0433
- **presentation_added** `` -> `tb_0433_c11`

### tb_0438
- **presentation_added** `` -> `tb_0438_c14`

### tb_0439
- **presentation_added** `` -> `tb_0439_c08`

### tb_0440
- **presentation_added** `` -> `tb_0440_c09`

### tb_0441
- **presentation_added** `` -> `tb_0441_c12`

### tb_0442
- **presentation_added** `` -> `tb_0442_c12`

### tb_0445
- **presentation_added** `` -> `tb_0445_c11`

### tb_0447
- **presentation_added** `` -> `tb_0447_c12`

### tb_0450
- **presentation_added** `` -> `tb_0450_c08`

### tb_0451
- **presentation_added** `` -> `tb_0451_c10`

### tb_0454
- **presentation_added** `` -> `tb_0454_c16`

### tb_0457
- **presentation_added** `` -> `tb_0457_c12`

### tb_0458
- **presentation_added** `` -> `tb_0458_c07`

### tb_0459
- **presentation_added** `` -> `tb_0459_c06`

### tb_0464
- **presentation_added** `` -> `tb_0464_c20`

### tb_0465
- **presentation_added** `` -> `tb_0465_c06`

### tb_0467
- **presentation_added** `` -> `tb_0467_c06`

### tb_0468
- **presentation_added** `` -> `tb_0468_c08`

### tb_0470
- **presentation_added** `` -> `tb_0470_c15`

### tb_0473
- **presentation_added** `` -> `tb_0473_c13`

### tb_0474
- **presentation_added** `` -> `tb_0474_c08`

### tb_0475
- **presentation_added** `` -> `tb_0475_c17`

### tb_0478
- **presentation_added** `` -> `tb_0478_c08`

### tb_0483
- **presentation_added** `` -> `tb_0483_c12`

### tb_0484
- **presentation_added** `` -> `tb_0484_c06`

### tb_0485
- **presentation_added** `` -> `tb_0485_c06`

### tb_0486
- **presentation_added** `` -> `tb_0486_c08`

### tb_0487
- **presentation_added** `` -> `tb_0487_c13`

### tb_0494
- **presentation_added** `` -> `tb_0494_c10`

### tb_0498
- **presentation_added** `` -> `tb_0498_c09`

### tb_0503
- **presentation_added** `` -> `tb_0503_c10`

### tb_0512
- **presentation_added** `` -> `tb_0512_c09`

### tb_0513
- **presentation_added** `` -> `tb_0513_c07`

### tb_0514
- **presentation_added** `` -> `tb_0514_c14`

### tb_0515
- **presentation_added** `` -> `tb_0515_c09`

### tb_0516
- **presentation_added** `` -> `tb_0516_c09`

### tb_0517
- **presentation_added** `` -> `tb_0517_c12`

### tb_0518
- **presentation_added** `` -> `tb_0518_c11`

### tb_0519
- **presentation_added** `` -> `tb_0519_c10`

### tb_0522
- **presentation_added** `` -> `tb_0522_c11`

### tb_0523
- **presentation_added** `` -> `tb_0523_c09`

### tb_0524
- **presentation_added** `` -> `tb_0524_c09`

### tb_0525
- **presentation_added** `` -> `tb_0525_c08`

### tb_0529
- **presentation_added** `` -> `tb_0529_c10`

### tb_0535
- **presentation_added** `` -> `tb_0535_c08`

### tb_0536
- **presentation_added** `` -> `tb_0536_c09`

### tb_0539
- **presentation_added** `` -> `tb_0539_c09`

### tb_0540
- **presentation_added** `` -> `tb_0540_c08`

### tb_0543
- **presentation_added** `` -> `tb_0543_c07`

### tb_0544
- **presentation_added** `` -> `tb_0544_c08`

### tb_0546
- **presentation_added** `` -> `tb_0546_c10`

### tb_0547
- **presentation_added** `` -> `tb_0547_c09`

### tb_0548
- **presentation_added** `` -> `tb_0548_c07`

### tb_0549
- **presentation_added** `` -> `tb_0549_c10`

### tb_0550
- **presentation_added** `` -> `tb_0550_c12`

### tb_0552
- **presentation_added** `` -> `tb_0552_c08`

### tb_0557
- **presentation_added** `` -> `tb_0557_c09`

### tb_0558
- **presentation_added** `` -> `tb_0558_c16`

### tb_0559
- **presentation_added** `` -> `tb_0559_c10`

### tb_0560
- **presentation_added** `` -> `tb_0560_c11`

### tb_0561
- **presentation_added** `` -> `tb_0561_c09`

### tb_0562
- **presentation_added** `` -> `tb_0562_c08`

### tb_0563
- **presentation_added** `` -> `tb_0563_c11`

### tb_0567
- **presentation_added** `` -> `tb_0567_c07`

### tb_0571
- **presentation_added** `` -> `tb_0571_c07`

### tb_0572
- **presentation_added** `` -> `tb_0572_c07`

### tb_0574
- **presentation_added** `` -> `tb_0574_c08`

### tb_0577
- **presentation_added** `` -> `tb_0577_c06`

### tb_0578
- **presentation_added** `` -> `tb_0578_c10`

### tb_0582
- **presentation_added** `` -> `tb_0582_c06`

### tb_0583
- **presentation_added** `` -> `tb_0583_c11`

### tb_0584
- **presentation_added** `` -> `tb_0584_c10`

### tb_0587
- **presentation_added** `` -> `tb_0587_c11`

### tb_0588
- **presentation_added** `` -> `tb_0588_c09`

### tb_0590
- **presentation_added** `` -> `tb_0590_c07`

### tb_0594
- **presentation_added** `` -> `tb_0594_c08`

### tb_0595
- **presentation_added** `` -> `tb_0595_c14`

### tb_0596
- **presentation_added** `` -> `tb_0596_c09`

### tb_0597
- **presentation_added** `` -> `tb_0597_c09`

### tb_0598
- **presentation_added** `` -> `tb_0598_c12`

### tb_0599
- **presentation_added** `` -> `tb_0599_c07`

### tb_0600
- **presentation_added** `` -> `tb_0600_c08`

### tb_0602
- **presentation_added** `` -> `tb_0602_c09`

### tb_0604
- **presentation_added** `` -> `tb_0604_c11`

### tb_0606
- **presentation_added** `` -> `tb_0606_c08`

### tb_0607
- **presentation_added** `` -> `tb_0607_c08`

### tb_0608
- **presentation_added** `` -> `tb_0608_c09`

### tb_0609
- **presentation_added** `` -> `tb_0609_c09`

### tb_0610
- **presentation_added** `` -> `tb_0610_c10`

### tb_0611
- **presentation_added** `` -> `tb_0611_c10`

### tb_0613
- **presentation_added** `` -> `tb_0613_c07`

### tb_0614
- **presentation_added** `` -> `tb_0614_c11`

### tb_0619
- **presentation_added** `` -> `tb_0619_c10`

### tb_0621
- **presentation_added** `` -> `tb_0621_c07`

### tb_0623
- **presentation_added** `` -> `tb_0623_c09`

### tb_0624
- **presentation_added** `` -> `tb_0624_c08`

### tb_0626
- **presentation_added** `` -> `tb_0626_c17`

### tb_0627
- **presentation_added** `` -> `tb_0627_c09`

### tb_0628
- **presentation_added** `` -> `tb_0628_c15`

### tb_0631
- **presentation_added** `` -> `tb_0631_c16`

### tb_0633
- **presentation_added** `` -> `tb_0633_c07`

### tb_0634
- **presentation_added** `` -> `tb_0634_c07`

### tb_0636
- **presentation_added** `` -> `tb_0636_c08`

### tb_0637
- **presentation_added** `` -> `tb_0637_c05`

### tb_0638
- **presentation_added** `` -> `tb_0638_c08`

### tb_0639
- **presentation_added** `` -> `tb_0639_c13`

### tb_0642
- **presentation_added** `` -> `tb_0642_c11`

### tb_0643
- **presentation_added** `` -> `tb_0643_c13`

### tb_0644
- **presentation_added** `` -> `tb_0644_c10`

### tb_0645
- **presentation_added** `` -> `tb_0645_c07`

### tb_0646
- **presentation_added** `` -> `tb_0646_c09`

### tb_0648
- **presentation_added** `` -> `tb_0648_c11`

### tb_0650
- **presentation_added** `` -> `tb_0650_c08`

### tb_0651
- **presentation_added** `` -> `tb_0651_c08`

### tb_0652
- **presentation_added** `` -> `tb_0652_c12`

### tb_0653
- **presentation_added** `` -> `tb_0653_c11`

### tb_0656
- **presentation_added** `` -> `tb_0656_c10`

### tb_0658
- **presentation_added** `` -> `tb_0658_c14`

### tb_0660
- **presentation_added** `` -> `tb_0660_c10`

### tb_0661
- **presentation_added** `` -> `tb_0661_c06`
