# Dr. SCI — the 5 skills, with real criteria

Browsing reference for the Dr. SCI q-matrix axis. **Understanding only** — no code loads this.
Every criterion below is quoted **verbatim from the shipped bank**.

Measured on the actual bank: **98,611 criteria** across **10,003 questions**, 1,429 per
subject over 7 subjects. Open-ended half only — the verifiable half has no criteria at all.
See [README.md](README.md) for scope, polarity and the difficulty inversion.

> **Assignment is by pattern rules over criterion text. Aggregate shares are reliable;
> individual row labels are not.** A hand-check of 16 random criteria found ~5 misassigned,
> mostly words used in a different sense. See `Hard cases`.


## The axis

| slug | plain meaning | % of items | % of positive weight | pure anchors |
|---|---|---|---|---|
| `factual_recall` | knows the fact | 46.6% | 44.3% | 31,845 |
| `mechanistic_reasoning` | can explain why | 41.8% | 46.8% | 24,261 |
| `quantitative` | can do the math | 22.5% | 25.3% | 5,270 |
| `presentation` | writes it clearly | 18.1% | 15.7% | 3,686 |
| `restraint` | doesn't overclaim | 6.8% | 3.8% | 1,440 |

Multi-hot, so shares exceed 100%. Loadings per criterion: 1→66,502, 2→28,906, 3→3,145, 4→58. No criterion is unmapped.

**On two of the names.** `restraint` is deliberately not called "calibration" — this repo
already uses that word for fitting the IRT model and for the pipeline `split` label, so a
skill by that name would be genuinely ambiguous. `presentation` is preferred over
"communication" because it says plainly that the criterion is about the *form* of the
answer rather than its content.


---

## `factual_recall` — knows the fact

Does the model **know the thing**? Satisfied by the answer containing a correct discrete claim — a fact, term, value, definition, classification. No explanation demanded.

**Lacking it:** does not know the science, or omits a required point.

**Coverage.** 45,965 items (46.6%); 31,845 load on this skill *only*.


| subject | cat | w | criterion (verbatim) |
|---|---|---|---|
| biology | Essential | 5 | Indicates that the risk rises (rather than falls or remains unchanged) as consumption of sugar-sweetened drinks increases. |
| chemistry | Essential | 4 | Correctly identifies copper(II) nitrate and water as the products of the reaction. |
| cs | Essential | 5 | Clearly states that a tautology is a logical statement that is true under all possible truth value assignments. |
| economics | Essential | 5 | Clearly states that the most appropriate tool is Open Market Operations (OMO) as the final answer. |
| medicine | Essential | 5 | States that ARF-deficient (knock-out) mice are more prone to developing tumors than wild-type mice. |
| physics | Essential | 5 | Correctly assigns a negative sign to the vertical velocity to reflect the electron’s acceleration in the direction opposite to the field. |
| science | Essential | 5 | Describes the relationships between the identified factors and the resulting symptom of steering wheel shaking. |
| biology | Important | 4 | Provides a detailed biochemical explanation that includes relevant metabolic pathways and signals, ensuring that key terms like 'ketone bodies' and 'gluconeogenesis' are accurately defined. |
| chemistry | Important | 4 | Accurately uses multiplicative prefixes like di- and tri- (as in 7,7-diethyl and 2,4,5-trimethyl) along with their locants. |
| cs | Important | 4 | Identifies benefits such as better escape from local minima or plateaus, improved convergence behavior, and enhanced exploration of the error landscape. |
| economics | Important | 4 | The response should emphasize the increase in net income ($130,000) rather than the increase in sales, as residual income evaluation primarily depends on incremental profit over the investment's cost of capital. |
| medicine | Important | 3 | Considers thumb sucking, tongue thrust, or abnormal swallowing patterns that may produce or maintain the space. |
| physics | Important | 4 | Describes how an increase in gravitational force affects the behavior of celestial bodies, including potential changes in orbital dynamics and system stability. |
| science | Important | 5 | The response must include detailed technical specifications and data, such as reactor parameters and propulsion performance metrics, to support the design proposal. |
| biology | Optional | 1 | Provides representative statistics or percentages illustrating negligible incidence reduction versus measurable duration decrease. |
| chemistry | Optional | 2 | Includes all three main uses—flotation collector, rubber accelerator, and chemical reagent—showing comprehensive coverage. |
| cs | Optional | 2 | Mentions that the proof works identically if induction is carried out on m instead of n, highlighting symmetry. |
| economics | Optional | 2 | Encourages writing specific, measurable goals (e.g., saving for college or a car) to guide allocation size and strategy. |
| medicine | Optional | 2 | Explicitly states that the appropriate dose depends on the specific clinical indication (hypertension, angina, rate control, etc.). |
| physics | Optional | 1 | Acknowledges that energy includes mass-energy equivalence (E=mc²) in relativistic contexts, showing conceptual breadth. |

---

## `mechanistic_reasoning` — can explain why

Can the model **show why**? Demands a causal chain, derivation, application of a principle, comparison or justification. Where a criterion demands both, reasoning takes precedence — explaining why subsumes stating what.

**Lacking it:** can recite facts but cannot connect them or say why something follows.

**Coverage.** 41,208 items (41.8%); 24,261 load on this skill *only*.


| subject | cat | w | criterion (verbatim) |
|---|---|---|---|
| biology | Essential | 5 | Explains the use of restriction enzymes to digest the plasmids in order to isolate fragments with compatible ends for ligation. |
| chemistry | Essential | 5 | Explains that velocity is inferred from the pressure difference created between the upstream-facing and downstream-facing ports. |
| cs | Essential | 5 | Correctly states that both evaluation strategies yield the same final result ((aa)(aa)) and explains the reason behind this consistency. |
| economics | Essential | 5 | Identifies (A) as a leftward shift of the cigarette demand curve because reduced health risk lowers consumers’ willingness to buy at every price. |
| medicine | Essential | 5 | Clearly explains that spatial resolution in imaging modalities is limited by system performance factors beyond just the radiation wavelength. |
| physics | Essential | 5 | Correctly derives and states that the box’s acceleration is given by a = g sin θ based on the balance of forces on the medallion. |
| science | Essential | 5 | Evaluates whether the response correctly integrates factors like mountain erosion rates, corrosion of titanium-stabilized stainless steel, and environmental exposure in estimating the legibility duration. |
| biology | Important | 4 | Mentions at least one mechanism (cell adhesion, ECM remodeling, migration guidance, or survival signaling) by which integrins influence metastatic steps. |
| chemistry | Important | 4 | Discusses how Gaussian bases form a systematically improvable, complete set when hierarchically enlarged (e.g., minimal→double-ζ→triple-ζ), and contrasts this with STO completeness in practice. |
| cs | Important | 4 | Details the lambda abstraction steps and two applications that are utilized in reducing the type judgment for product types. |
| economics | Important | 4 | Provides a critical analysis of the responsibility-accounting system, including an evaluation of whether plant managers should be held responsible for both costs and profits. |
| medicine | Important | 2 | Notes that dehydration, shock, or renal failure can markedly decrease urine (oliguria/anuria) and sweat production, demonstrating balanced coverage of both increases and decreases. |
| physics | Important | 4 | The response must correctly apply vector subtraction to show that the change in momentum is the difference between the final and initial momentum vectors. |
| science | Important | 4 | Explains how each issue, such as oil leaks or excessive oil in the crankcase, leads to the described symptoms like blue smoke and loss of performance. |
| biology | Optional | 2 | Briefly explains how adrenal cortex overactivity or underactivity causes the respective hormone imbalance. |
| chemistry | Optional | 2 | The response may enhance understanding by briefly comparing the derived density with the provided reference values (1.222 g/mL for w/w or 1.187 g/mL for w/v). |
| cs | Optional | 2 | Indicates that heapsort is not stable and briefly explains why elements with equal keys may reorder. |
| economics | Optional | 3 | Provides a concrete, real-life example that demonstrates the application of one or more methods in a manufacturing setting, thereby enhancing the practical understanding of the concepts. |
| medicine | Optional | 2 | Briefly references expected pharmacological actions (e.g., anti-inflammatory saponins or antimicrobial flavonoids) to support the indications. |
| physics | Optional | 2 | May include direct references to the diagram to support reasoning and illustrate how the diagram guides the identification of current directions. |

---

## `quantitative` — can do the math

Can the model **run the numbers**? Formula selection, substitution, computation, units, conversions, precision, sign conventions.

**Lacking it:** understands the concept but botches the arithmetic or drops units.

**Coverage.** 22,210 items (22.5%); 5,270 load on this skill *only*.


| subject | cat | w | criterion (verbatim) |
|---|---|---|---|
| biology | Essential | 5 | Accurately converts energy units from kcal to joules or moles as needed and relates them to ATP and glucose energetics. |
| chemistry | Essential | 5 | The answer must provide the correct balanced chemical equation for the reaction, showing the stoichiometric relationship among reactants and products. |
| cs | Essential | 5 | Proves that the maintained remainder equals b mod a at every step, concluding that a divides b iff the final remainder is zero. |
| economics | Essential | 5 | Properly determines the average total cost (ATC) by summing the computed AFC and AVC for each output level. |
| medicine | Essential | 5 | Gives the accepted one-year risks (≈30 % for recent rest pain < 48 h, ≈12 % for rest pain > 48 h/new onset, and ≈42 % for post-infarction angina). |
| physics | Essential | 5 | Converts units properly by changing the charge from nanoCoulombs to Coulombs and distances from millimeters to meters. |
| science | Essential | 4 | Must use correct astronomical terminology such as 'orbit clearing', 'celestial body', and 'natural satellite', ensuring precision in the explanation. |
| biology | Important | 3 | Uses correct units (mg kg⁻¹ or % dry weight) and no contradictory numbers. |
| chemistry | Important | 4 | Checks that the units cancel appropriately, ensuring that the result is in grams. |
| cs | Important | 3 | Uses correct graph-theoretic terms such as edge, weight, spanning tree, cycle, path, and optimality without mislabeling. |
| economics | Important | 4 | Uses correct currency pair notation and units, confirming that the strike prices are correctly understood in the context of USD/CAD. |
| medicine | Important | 2 | Uses precise medical terms like "arterial thromboembolism," "cerebral ischemia," or "cardioembolic stroke" rather than vague phrases such as "clot moving around." |
| physics | Important | 2 | Keeps all distances in metres, masses in kilograms, and energies in joules throughout the calculation. |
| science | Important | 4 | The answer should illustrate that the river and aquifer exchange is dynamic, with potential for the river to either recharge the aquifer or receive groundwater depending on local conditions. |
| biology | Optional | 2 | Provides context regarding variability among individuals or additional factors that could influence brain energy use. |
| chemistry | Optional | 1 | Supplies representative empirical equations or coefficient ranges from L’Heureux & Long (e.g., s_u = a·V_s^b) to illustrate the quantified trends. |
| cs | Optional | 2 | Provides a concrete numerical example showing calculated radius, speed, and steering angle for a sample intersection to illustrate application. |
| economics | Optional | 3 | Ensures that the response consistently uses standard deviation units throughout the answer to avoid any ambiguity regarding measurement scales. |
| medicine | Optional | 2 | Emphasizes patient values, comorbidities, life expectancy, and quality-of-life considerations in individualized therapy planning. |
| physics | Optional | 1 | Provides illustrative examples (e.g., black-hole entropy, digital universe estimates) to make abstract limits more concrete. |

---

## `presentation` — writes it clearly

Is the answer **well presented**? Clear, logically ordered, appropriately concise, easy to follow. About the FORM of the answer, not its content.

**Lacking it:** correct but rambling, disorganised or padded.

**Coverage.** 17,857 items (18.1%); 3,686 load on this skill *only*.


| subject | cat | w | criterion (verbatim) |
|---|---|---|---|
| biology | Essential | 5 | Explicitly excludes skeletal muscle from the answer since it is voluntary, ensuring clarity of the response. |
| chemistry | Essential | 5 | Draws both major Lewis resonance forms of NO₂, each showing one N=O double bond, one N–O single bond, and the unpaired electron on nitrogen (odd-electron radical). |
| cs | Essential | 5 | Provides a concrete example of a recursive method that illustrates the concept in practice, ensuring the example is clear and relevant. |
| economics | Essential | 5 | Accurately categorizes assets into current and non-current groups based on the trial balance information. |
| medicine | Essential | 5 | Ends with a clear and concise summary that ties together the advantages and necessary considerations for deploying portable input devices. |
| physics | Essential | 5 | Ends with a clear and direct conclusion regarding the feasibility of creating a photon bubble and its implications on the spaceship's mass. |
| science | Essential | 5 | Details the changes in the steel's microstructure, such as the decomposition of martensite and formation of carbides, which are pivotal for achieving the desired combination of hardness and toughness. |
| biology | Important | 4 | Details the overall process of generating a genetic fingerprint by combining PCR, electrophoresis, and restriction enzyme digestion. |
| chemistry | Important | 4 | Combines the two components in one coherent phrase such as “DNA with an RNA primer at the 5′ end.” |
| cs | Important | 4 | Provides a concise explanation for each design principle (Fitt’s Law, Consistency, Forgiveness, and Minimalism) and how each is applied to enhance the interface. |
| economics | Important | 4 | Uses a supply and demand graph with properly labeled axes (interest rate and quantity) and curves to visually support the explanation. |
| medicine | Important | 3 | Separates information explicitly into morphology, biochemistry, and ecology so readers can easily follow each aspect. |
| physics | Important | 4 | The response should synthesize the roles of gravity, dark matter, gas dynamics, and collisional behavior into a coherent explanation of galaxy formation. |
| science | Important | 4 | The response should be organized and clear, providing sufficient detail and explanation for each cleaning technique and associated risk. |
| biology | Optional | 2 | The response may utilize an interval or range format (e.g., '29.5 to 35 inches') to clearly communicate the measurement details. |
| chemistry | Optional | 1 | Delivers all required content without unnecessary digressions, keeping the response brief and focused. |
| cs | Optional | 3 | Uses clear, well-labeled diagrams for both key creation processes to enhance overall clarity and understanding. |
| economics | Optional | 2 | Communicates the key points succinctly without unnecessary filler, making the argument both accessible and efficient. |
| medicine | Optional | 1 | Delivers the answer succinctly without unnecessary digressions, ideally within a short paragraph. |
| physics | Optional | 3 | The answer is presented in a clear and well-organized manner, with logical sequencing and proper technical language. |

---

## `restraint` — doesn't overclaim

Does the model **avoid overclaiming**? Hedges where evidence is mixed, names assumptions, respects scope, does not present contested things as settled. Catches many `Pitfall` criteria.

**Lacking it:** overstates, over-generalises, asserts more than the evidence supports.

**Coverage.** 6,741 items (6.8%); 1,440 load on this skill *only*.


| subject | cat | w | criterion (verbatim) |
|---|---|---|---|
| biology | Essential | 4 | Gives Sudan Black B as the lone answer without listing additional mutually exclusive dyes or expressing uncertainty (e.g., avoids "Sudan Black B or Nile Red"). |
| chemistry | Essential | 5 | Clarifies that the chiral BINAP (or related) ligand creates an asymmetric environment that directs the facial approach of the ketone, thus determining the product’s absolute configuration. |
| cs | Essential | 5 | It must explicitly detail the theoretical limitations, such as the impact of signal noise and the human eye/brain's capacity to reconstruct images from damaged data. |
| economics | Essential | 5 | Uses correct economic terminology and definitions for costs like shoeleather and menu costs without oversimplifying or misrepresenting them. |
| medicine | Essential | 4 | Warns that any internal use of White Lily should be undertaken cautiously and preferably under qualified professional supervision. |
| physics | Essential | 5 | Maintains rigorous mathematical treatment by ensuring no steps are omitted and all intermediate results and assumptions are clearly stated. |
| science | Essential | 5 | Provides an estimated total volume of Earth's water by incorporating assumptions about the average ocean depth and the percentage of the Earth's surface covered by water. |
| biology | Important | 4 | Explicitly anchors the term within dermatology by noting its use to characterize cutaneous lesions or dermal architecture rather than unrelated organ systems. |
| chemistry | Important | 3 | Clarifies that Le Chatelier’s principle only changes molecule numbers when the chemical reaction actually involves a different count of gaseous moles on each side. |
| cs | Important | 3 | Explicitly clarifies any assumptions or conditions under which the repeated use of universal instantiation is applied. |
| economics | Important | 4 | Clearly discloses any assumptions, such as the treatment of the discount rates and the timing of coupon payments and reinvestments, to ensure complete transparency in the method used. |
| medicine | Important | 3 | Instructs that children should only receive dill under guidance from a qualified healthcare provider or herbalist. |
| physics | Important | 4 | The response should detail how momentum conservation is used in tandem with energy conservation, even given the heavy nucleus assumption. |
| science | Important | 3 | The response emphasizes the need to handle hot coolant and overheated engine parts carefully to prevent burns or further damage. |
| biology | Optional | 1 | Reminds users to wear gloves, goggles, and avoid inhaling dust when handling the solid dye and acids/bases for pH adjustment. |
| chemistry | Optional | 1 | Uses standard electron-configuration notation and correct quantum-mechanical terminology without introducing informal or misleading terms. |
| cs | Optional | 3 | Provides explicit assumptions such as data centering and standardization that are necessary for the claim of a zero residual matrix to hold true. |
| economics | Optional | 3 | Provides a balanced view that while research and patience are critical, market uncertainties persist and no approach guarantees success. |
| medicine | Optional | 2 | Advises against prescribing topical anesthetics for home use and recommends alternative pain control plus return precautions. |
| physics | Optional | 1 | Optionally considers that experimental errors or calibration issues might also play a role, even if briefly, in the unexpected measurement outcome. |

---

## Criteria loading on more than one skill

| skills | subject | cat | criterion (verbatim) |
|---|---|---|---|
| `factual_recall + presentation` | medicine | Optional | Presents information in a concise, well-structured, and grammatically sound manner without unnecessary digressions. |
| `factual_recall + presentation` | biology | Optional | Presents the answer in clear, concise language without unnecessary jargon or digressions. |
| `factual_recall + presentation` | chemistry | Pitfall | Describes Gran titration as plotting pH versus volume without the 10^{−pH} (or similar) transformation, showing misunderstanding of the method. |
| `factual_recall + presentation` | cs | Pitfall | Should not include unnecessary complexities or extraneous components that obscure the primary graph representation of material movement. |
| `mechanistic_reasoning + quantitative` | economics | Essential | The answer must incorporate the use of the natural logarithm of 2 (ln 2) in its calculation, demonstrating a sound understanding of the doubling time formula ln(2)/growth rate. |
| `factual_recall + presentation` | science | Essential | Clearly states multiple definitions of dimension, including those based on vector space basis, transcendence basis, and chain lengths of substructures, as well as relevant definitions in physics. |
| `mechanistic_reasoning + quantitative` | biology | Pitfall | The response must avoid offering only numerical values, such as stating '2/3', without explaining the underlying optical mechanisms that justify this figure. |
| `mechanistic_reasoning + quantitative` | economics | Essential | Verifies that the response correctly calculates the new total sales revenue by applying the per-unit sales price to 3,400 units, arriving at $268,600. |
| `mechanistic_reasoning + quantitative` | science | Important | Correctly derives and explains the percentage of backflow moves with supporting calculations that reflect the provided numerical result. |
| `mechanistic_reasoning + quantitative` | economics | Optional | Presents the reasoning process in a logical sequence, starting from individual elements (contribution margin and fixed expenses) to the final conclusion. |
| `mechanistic_reasoning + presentation + restraint` | science | Optional | Organizes the response in a logical sequence from data smoothing to derivative extraction and finally to uncertainty estimation. |
| `mechanistic_reasoning + quantitative` | cs | Important | Analyzes the balance between higher precision (or convergence guarantees) and practical concerns such as time complexity, memory usage, or energy cost in computational mathematics. |
| `mechanistic_reasoning + quantitative` | chemistry | Essential | Verifies that the final numerical answer is rounded to the nearest hundredth as specified in the question. |
| `factual_recall + quantitative` | physics | Pitfall | Gives a range outside 15 – 20 %, such as 5 – 10 % or over 25 %, indicating misunderstanding of standard values. |

---

## Inverted (`Pitfall`) criteria — 17% of the bank

These name a mistake to AVOID. Scored as: did the response avoid it? The original negative
weight is kept in `source_weight`.

| skills | subject | w | criterion (verbatim) |
|---|---|---|---|
| `factual_recall` | economics | -2 | Omits or incorrectly treats the one-time $40 machine expense, leading to understated costs. |
| `mechanistic_reasoning` | chemistry | -2 | Ensures that the coefficients from the reaction are correctly interpreted and used, avoiding common errors such as using a wrong stoichiometric factor. |
| `factual_recall` | biology | -2 | Does not mistakenly attribute the lower blood level to other factors such as fat content or metabolic differences instead of water. |
| `factual_recall` | economics | -2 | The answer must avoid mixing entries for payroll taxes (such as FICA or unemployment taxes) with those for employee benefits, which is a common error. |
| `factual_recall` | chemistry | -1 | The response should avoid confusing peroxides with related species such as superoxides and must not omit the distinction that dioxides do not have the O2^2- ion. |
| `restraint` | biology | -2 | Claims that the breeding season lasts several weeks or months rather than only a handful of nights. |
| `factual_recall` | cs | -2 | Treats the grid as axis-aligned or uses standard Cartesian cell indexing without adapting to the non-orthogonal basis, leading to incorrect geometry. |
| `mechanistic_reasoning` | science | -2 | Mentions fuel and spark but neglects to evaluate compression or mechanical timing, leaving a critical cause unaddressed. |
| `mechanistic_reasoning + quantitative` | chemistry | -1 | Avoids the common error of reporting an extra significant figure in the final answer by correctly applying significant figure rules throughout the multi-step calculation. |
| `factual_recall` | biology | -1 | Misattributes the findings to chronic obstructive pulmonary disease or emphysema despite clear lung fields and absence of hyperinflation. |

---

## Hard cases


**The recall / reasoning border is fuzzy for roughly a third of items**

> "Explains that alcohol increases hepatic VLDL synthesis, thereby raising serum triglycerides."

Knowing the mechanism and reasoning through it are one act here. The rule gives reasoning precedence -- defensible, but arbitrary at the margin. Expect these two theta to correlate more than the others.


**Words used in a different sense break the pattern rules**

> "...clarifying that the union (not the intersection) is implemented."

'Clarifying' here means stating precisely, not writing clearly -- but it can trigger presentation. Likewise 'presents' and 'defines' falsely suggest recall, and product numbers like .357 falsely suggest quantitative.


**`restraint` is small and lives in the cheapest criteria**

> 6.8% of items but only 3.8% of positive weight

The most likely of the five to be underpowered despite 1,440 pure anchors. If it proves so, fold it into mechanistic_reasoning.


**Expect the content dimensions to collapse**

> on TutorBench's 82-model matrix: content~quantitative r=0.943, and each ~0.97-0.99 with total score

Different benchmark, so a hypothesis rather than a result -- but if factual_recall and mechanistic_reasoning come back correlated above 0.90 on Dr. SCI, merge them. Relabelling is free; it does not require re-judging.


## Pre-registered kill criteria

Decided in advance so a skill is dropped on evidence rather than after the fact:

- latent correlation with another skill **> 0.90** -> merge (`calibrate_mirt.py --collapse`)
- theta correlates **> 0.85** with one subject's pass rate -> subject proxy, drop
- a weak base model (<=1B) lands in the top quartile -> artifact, drop
- pass rate **>=0.95 or <=0.05** across models -> no variance, no information

