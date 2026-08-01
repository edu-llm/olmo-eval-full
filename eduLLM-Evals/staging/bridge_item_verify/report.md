# Bridge adversarial verification — trying to disqualify what we kept

`openai-group/gpt-5.5`, 3 independent attack lenses per scenario (239/239 scenarios with a complete set). Each attempt is told to assume the item is unfit and prove it, resolving uncertainty toward disqualification. A scenario is suspect on a majority (>=2 of 3), since an adversarial prompt will always manufacture some objection.


## Headline


| | scenarios | share |
|---|---:|---:|
| survived all 3 attacks | 93 | 39% |
| 1 of 3 disqualified (noise floor) | 79 | 33% |
| **suspect (>=2 of 3)** | **67** | **28%** |
| unanimous (3 of 3) | 18 | 8% |


## Which lens disqualifies most


A lens firing far more than the others is likelier to be over-eager than right.


| lens | disqualified | of complete |
|---|---:|---:|
| `answerability` | 30 | 13% |
| `criteria_fit` | 82 | 34% |
| `gold_soundness` | 119 | 50% |


## Severity among suspect items


| worst severity claimed | scenarios |
|---|---:|
| fatal | 64 |
| serious | 3 |


## Unanimously disqualified — 18


**bridge_0009** · student said `rhombus` · fatal
- *answerability*: The visible question is too underspecified/ambiguous to determine a unique correct answer: “What is the name of quadrilateral having the length bigger than width?” Many quadrilaterals can be longer than they are wide, and the prom
- *criteria_fit*: The underlying tutor question is mathematically underdetermined: “What is the name of quadrilateral having the length bigger than width?” does not uniquely specify a quadrilateral type. Many quadrilaterals could have a greater len
- *gold_soundness*: The tutoring premise is ill-posed: the prompt asks for “the name of quadrilateral having the length bigger than width,” but that description does not uniquely define a quadrilateral. Many quadrilaterals can have an overall length 

**bridge_0028** · student said `she would have 1,200` · fatal
- *answerability*: The item lacks the original problem or number being discussed, so the tutor cannot determine a specific math error or misconception. The only visible prompt is “Where is the tens place?” and the student answers “she would have 1,2
- *criteria_fit*: The attached critical conceptual-error criterion does not fit the item. The visible exchange shows no discernible wrong place-value reasoning: the tutor asks, "Where is the tens place?" and the student replies, "she would have 1,2
- *gold_soundness*: The transcript is too truncated/incoherent to establish a math error or misconception. The tutor asks only, “Where is the tens place?” and the student replies, “she would have 1,200.” The gold key then affirms the student’s conten

**bridge_0104** · student said `done` · fatal
- *answerability*: The student's final turn is not a mathematical answer to check. The tutor asks, "What is a rectangle?" and the student replies only "done." This is just a completion/acknowledgment, not a definition or reasoning error, so a tutor 
- *criteria_fit*: There is no student mathematical answer or visible mistake to diagnose: the tutor asks, "What is a rectangle?" and the student's entire final turn is "done". Thus criteria about addressing "the underlying misconception or faulty r
- *gold_soundness*: The gold key gives a mathematically incorrect/incomplete definition: it says, “A rectangle has two equal sides and two equal angles. That's the difference between rectangles and other quadrilaterals.” A rectangle is defined by fou

**bridge_0108** · student said `six` · fatal
- *answerability*: The visible problem statement is malformed/ambiguous: it asks, "What is the place value of ten in 262,956?" There is no digit "ten" in the number, and "ten" is not the place name; the likely intended question is about the "tens" p
- *criteria_fit*: The prompt itself is malformed/ambiguous: the tutor asks, "What is the place value of ten in 262,956?" There is no digit "ten" in the number, and if the intended question is about the tens place, the expected answer could be the d
- *gold_soundness*: The tutor's original question is malformed/ambiguous: “What is the place value of ten in 262,956?” There is no digit “ten” in the number, and “place value” would normally ask for a value/place, not simply the digit. The gold key a

**bridge_0114** · student said `9` · fatal
- *answerability*: The original problem is absent, and the student's answer is only a bare number: "[student] 9". From the visible text, a tutor cannot determine what the final answer was supposed to be or why 9 is wrong without relying on the exper
- *criteria_fit*: Criterion F1 is beside the point for this item. It requires: “The response treats the fraction or mixed number as a quantity -- referring to its size, or to what it is a part of,” but the visible transcript contains no fraction/mi
- *gold_soundness*: The visible transcript omits the actual fraction/problem, so the student's final answer cannot be identified as wrong. The only math context is: tutor says “The 1 is in numerator part” and asks “What is our final answer?”, then th

**bridge_0133** · student said `ok i thought that we turn the whole number m` · fatal
- *answerability*: The student's final turn is not a concrete wrong answer; it is an acknowledgment plus an ambiguous statement: "ok i thought that we turn the whole number more yes i do." A tutor cannot name a specific mathematical error from this 
- *criteria_fit*: The final student turn contains no concrete mathematical answer or identifiable error to correct: "ok i thought that we turn the whole number more yes  i do". The attached critical criterion I1 presupposes a specific imprecise mat
- *gold_soundness*: There is no concrete student math error to correct in the final turn. The student says only, "ok i thought that we turn the whole number more yes i do," which is vague but not an identifiable wrong answer or mathematical mistake. 

**bridge_0144** · student said `4` · fatal
- *answerability*: The visible prompt is ambiguous/contradicts the gold key about which digit changes. It asks, “What will the 4 in 346 become when we borrow from the hundreds column?” Borrowing from the hundreds column into the tens column would ma
- *criteria_fit*: The prompt and gold key conflict mathematically, so the student's error is not well-defined. The tutor asks, "What will the 4 in 346 become when we borrow from the hundreds column?" In 346, the 4 is the tens digit; borrowing from 
- *gold_soundness*: The gold key gives the wrong regrouping guidance. The tutor asks, “What will the 4 in 346 become when we borrow from the hundreds column?” In 346, the 4 is the tens digit; borrowing from the hundreds column would make it 14 tens, 

**bridge_0223** · student said `i dont no` · fatal
- *answerability*: The student did not make a specific math error to diagnose; they only expressed lack of knowledge: tutor asks, “Why is it important to check if there is any unnecessary information in a word problem?” and student replies, “i dont 
- *criteria_fit*: The attached operations criterion is beside the point and cannot be applied: O1 requires that the response "connects the operation to its meaning or to the problem situation," but the visible exchange contains no arithmetic operat
- *gold_soundness*: The gold key gives the opposite of the intended guidance due to a missing negation: “It's so that we know use the wrong information when calculating the answer.” This says checking unnecessary information helps us use the wrong in

**bridge_0254** · student said `b` · fatal
- *answerability*: The student's final answer is only a bare label: after the tutor asks, "What are the factors of -8?" the student replies "b". No answer choices or meaning for "b" are visible, so the tutor would have to guess what the student sele
- *criteria_fit*: The attached right-idea criteria require a correct/partial idea to build on, but the student's entire final answer is just "b" to the question "What are the factors of -8?" There is no visible mathematical reasoning or partial und
- *gold_soundness*: The gold key gives a mathematically impossible/inaccurate prompt for the actual question. The tutor asks, “What are the factors of -8?”, but the expert reply says factors are “whole numbers that multiply to equal -8.” Whole number

**bridge_0262** · student said `300` · fatal
- *answerability*: The visible text never states what the graph's two quantities/axes are: it only says “unit rate,” “one point on the line,” and “coordinates.” Yet criterion Z1 requires the tutor to “identify which two quantities the rate or ratio 
- *criteria_fit*: Criterion Z1 is unanswerable for this item: it requires that the response “identifies which two quantities the rate or ratio compares,” but the transcript provides no quantities or axis labels—only “unit rate,” “one point on the l
- *gold_soundness*: The attached criterion requires information that is absent from the transcript: Z1 says the response should “identify which two quantities the rate or ratio compares,” but the visible conversation only says “selecting one point on

**bridge_0342** · student said `hexagon` · fatal
- *answerability*: The gold key contains a false mathematical statement: "The prefix for 5 sides is hexa." Hexa- means 6, not 5; a 5-sided shape is a pentagon, using penta-.
- *criteria_fit*: The gold key is mathematically wrong: it says, "The prefix for 5 sides is hexa." But "hexa-" means six; a 5-sided shape is a pentagon, from "penta-." This would guide the tutor/judge toward the student's wrong answer "hexagon."
- *gold_soundness*: The gold key gives mathematically incorrect guidance: after the student says a 5-sided shape is a “hexagon,” the expert reply says, “The prefix for 5 sides is hexa.” But “hexa-” means 6; the 5-sided shape is a pentagon, with prefi

**bridge_0419** · student said `nhw` · fatal
- *answerability*: The problem statement is absent from the visible transcript, so the tutor cannot know what kind of answer is required. The only visible final response is "[student] nhw" after "[tutor] What is the final answer?", and the gold key'
- *criteria_fit*: The tutor-visible transcript contains no math prompt or target concept—only “[tutor] What is the final answer? [student] nhw”—yet the gold key expects the tutor to ask about “a number that is a multiple of 10.” That requirement is
- *gold_soundness*: The transcript is truncated/incoherent and contains no actual math prompt or prior answer: it only shows “[tutor] What is the final answer? [student] nhw.” The student’s final turn “nhw” is just a nonsensical string, not a legible

**bridge_0487** · student said `it is 11?` · fatal
- *answerability*: The gold expert reply addresses a different subtraction problem than the visible transcript. The tutor asks, “What is the value of 15 - 6?” and the student answers “it is 11?”, but the expert reply says, “Let's do 24 - 7 on the wh
- *criteria_fit*: The item is internally inconsistent: the transcript asks the student to solve “15 - 6” (“What is the value of 15 - 6?”), but the gold tutor reply switches to a different problem: “Let's do 24 - 7 on the whiteboard...”. This makes 
- *gold_soundness*: The gold key addresses a different subtraction problem than the transcript. The tutor asks, “What is the value of 15 - 6?” and the student answers “it is 11?”, but the expert reply says, “Let’s do 24 - 7 on the whiteboard...” This

**bridge_0508** · student said `8: 15` · fatal
- *answerability*: The visible conversation is internally inconsistent about what the student is answering. The tutor asks, “How long is Luciano's train ride?” and the prior line already gives that answer: “His train ride takes 45 minutes.” The stud
- *criteria_fit*: The attached criteria assume the student's issue is only an imprecision/label gap, but the visible transcript does not support that. The tutor asks, “How long is Luciano's train ride?” after stating “His train ride takes 45 minute
- *gold_soundness*: The gold key addresses a different question and affirms a wrong response. The tutor asked, "How long is Luciano's train ride?" and the transcript already states, "His train ride takes 45 minutes." The student answered "8: 15," whi

**bridge_0565** · student said `im done` · fatal
- *answerability*: The student's final turn is not a mathematical answer to evaluate; it is just a status statement: after the tutor asks, “What is the value of  6 + 8?”, the student says only “im done.” There is no visible answer or specific mistak
- *criteria_fit*: The final student turn contains no mathematical answer or precision issue: "im done". But the critical criterion assumes an imprecise mathematical answer needing refinement: "when the issue is a precision gap (such as units, label
- *gold_soundness*: The transcript is missing the student's mathematical answer/mistake. The tutor asks, "What is the value of  6 + 8?" and the student's final turn is only "im done". The gold key then says, "Can you explain how you got your answer?"

**bridge_0607** · student said `8: 15` · fatal
- *answerability*: The transcript gives conflicting targets, so the student's final answer cannot be judged unambiguously. The immediate tutor question is “How long is Luciano's train ride?” and the transcript already states “His train ride takes 45
- *criteria_fit*: The attached imprecision criteria do not fit the actual student error. The tutor asks, “How long is Luciano's train ride?” after already stating “His train ride takes 45 minutes,” but the student answers “8: 15,” which is a clock 
- *gold_soundness*: The gold key affirms the student's answer even though it does not answer the tutor's actual question. The tutor asks, "How long is Luciano's train ride?" and the transcript already states the ride "takes 45 minutes." The student a

**bridge_0623** · student said `how long it is` · fatal
- *answerability*: The visible student response is not actually wrong on its face. The tutor asks, “What is Measuring length?” and the student answers, “how long it is,” which is a reasonable definition of measuring length at this level. The item re
- *criteria_fit*: The attached criterion U1 is beside the point and cannot be applied: it is for “topic_domain:measurement_conversion” and says the response must not “confuse or omit the units involved where the unit is what the problem turns on.” 
- *gold_soundness*: The gold key’s suggested correction is mathematically imprecise/incomplete for length measurement: it says, “a number that tells how long it is,” but a length measurement requires a number with a unit. This directly conflicts with

**bridge_0639** · student said `4` · fatal
- *answerability*: The student's final answer is not a clear mistake. The tutor asked, “What number will go in the ‘part?’” and the student answered “4.” The gold key explicitly treats this as correct: “Yes 4 is the part of 100.” Since the item requ
- *criteria_fit*: The item is supposed to be cut at a student mistake, but the supplied gold key explicitly treats the student's final answer as correct: the student answers "4" to "What number will go in the 'part?'" and the expert reply says, "Ye
- *gold_soundness*: The gold key affirms the student's wrong answer instead of correcting it. The tutor asks, "What number will go in the \"part?\"" for the problem "27 is 4% of what number?" In a percent proportion, the part is 27 and the whole is u


## Majority (2 of 3) — 49

- **bridge_0042** `11:01` — criteria_fit: The attached criterion U1 is beside the point: "[U1 · topic_domain:measurement_conversion · standard] The resp · gold_soundness: The gold key is not a usable tutor response because it refers to an absent whiteboard action rather than givin
- **bridge_0043** `1:55` — criteria_fit: The attached unit criterion is beside the point. The actual prompt asks for a clock numeral: “Which number is  · gold_soundness: The gold key is not a valid next tutor turn: it includes what appears to be the student's answer after the tut
- **bridge_0051** `30ft` — criteria_fit: The critical imprecision criterion does not fit the actual student mistake. I1 says the response should not tr · gold_soundness: The item's tutoring premise/criteria misclassify the student's error as an imprecision gap: I1 says the respon
- **bridge_0053** `i did it` — answerability: The student's final turn contains no mathematical answer to evaluate: after the tutor asks, "What is the unit  · gold_soundness: There is no visible mathematical answer or mistake to tutor. The tutor asks, “What is the unit for the volume?
- **bridge_0058** `prime` — criteria_fit: The attached criterion N1 is about place value, but the item is about prime vs. composite numbers. The transcr · gold_soundness: The gold key contains a mathematical reversal that would mislead the judge/student: it asks, “Is 3 divisible b
- **bridge_0098** `13` — criteria_fit: The gold key introduces unsupported numbers and an unsupported diagnosis: it says, “It looks like you subtract · gold_soundness: The gold key diagnoses a slip involving numbers that never appear in the transcript: the tutor asks, “What is 
- **bridge_0101** `if i haf 3 in 1 bag but  he  ole h` — answerability: The student does not give a legible answer or working to the visible question “What would be the value of 4 ti · gold_soundness: The student's final turn is too incoherent/nonresponsive to establish a specific math error to tutor. After be
- **bridge_0107** `11` — criteria_fit: The transcript gives contradictory operation/context, making the intended subtraction and any required connect · gold_soundness: The transcript is internally inconsistent because the tutor states the subtraction in the wrong order: “We nee
- **bridge_0131** `it has all of the sides equal` — criteria_fit: The attached criteria assume a 'right_idea' error with usable partial understanding: R1 says the response shou · gold_soundness: The item is mislabeled as a “right_idea” error and the criteria require building on a correct part of the stud
- **bridge_0157** `it  goes on for every` — answerability: The student has not made a checkable math error. The tutor asks an open-ended question, “What do you know abou · gold_soundness: The student's final turn is incoherent/truncated rather than a math mistake: "it  goes on for every". There is
- **bridge_0159** `by multyplying` — criteria_fit: The gold key shown to the judge is mathematically unsound and conflicts with the language criterion: it says,  · gold_soundness: The gold key is mathematically incorrect/misleading: it says, "the formula to find the speed of the train is m
- **bridge_0163** `it  goes on for every` — answerability: There is no checkable math task or specific wrong answer. The tutor only asks an open-ended prompt, "What do y · gold_soundness: The student's final turn is not a clear math error but an incomplete/truncated fragment: "it  goes on for ever
- **bridge_0176** `30ft` — criteria_fit: The critical criterion is mismatched to the actual error. It labels the issue as an imprecision/precision gap: · gold_soundness: The gold key responds to a different/unsupported task and invents a possible numerical answer. The visible pro
- **bridge_0202** `200` — criteria_fit: The gold key corrects the wrong digit. The prompt asks, “What is the place value of 2 in 521?” and the student · gold_soundness: The gold key does not answer the tutor’s question. The tutor asked, “What is the place value of 2 in 521?” and
- **bridge_0203** `no have 4 angles` — criteria_fit: Criterion V2 is not fit for this item. It requires the tutor response to “name the relevant figure, attribute  · gold_soundness: The gold key affirms the student's erroneous/non-answer instead of correcting it: the tutor asks, “What are de
- **bridge_0209** `yes` — criteria_fit: The attached 'right_idea' criteria require assessing whether the tutor builds on a correct part of the student · gold_soundness: The attached criteria assume the student has shown a partially correct strategy, but the transcript contains n
- **bridge_0212** `7 times 5=35` — answerability: The visible tutor prompt asks for “an example of an inverse operation,” but the gold key says the expected cor · gold_soundness: The gold key is mathematically wrong for the tutor's question. The tutor asks for "an example of an inverse op
- **bridge_0214** `6` — answerability: The student’s final response is ambiguous because it follows a different, yes/no question. The tutor first ask · gold_soundness: The transcript is ambiguous/incoherent at the cut: the tutor’s immediately preceding question is a yes/no ques
- **bridge_0243** `yes` — criteria_fit: The attached right_idea criteria require responding to a visible correct part/partial understanding, but the s · gold_soundness: The attached criteria assume a visible 'right idea' or partial understanding to build on, but the transcript s
- **bridge_0248** `9x5 I think` — criteria_fit: The gold key contains a mathematically false statement: “The formula will have only letters.” The area formula · gold_soundness: The gold key gives mathematically false guidance: it says, "The formula will have only letters." But the area 
- **bridge_0252** `yes` — criteria_fit: The attached right_idea criteria require building on an identifiable correct part of the student's thinking: R · gold_soundness: The gold key appears to violate the attached operations criterion. O1 requires: "The response connects the ope
- **bridge_0284** `54` — criteria_fit: The right_idea criteria require responding to and building on a visible correct part of the student's thinking · gold_soundness: The gold key gives a different subtraction problem instead of guiding the student to fix the asked one. The tu
- **bridge_0286** `3` — criteria_fit: The attached right-idea criteria are not answerable for this transcript. The student gives only an unsupported · gold_soundness: The item is mislabeled/rubriced as a “right_idea” error requiring the tutor to build on a correct part of the 
- **bridge_0300** `yas` — criteria_fit: The attached right-idea criteria require a visible correct part or partial understanding, but the transcript c · gold_soundness: The gold key does not actually identify or correct the student's math error. The student answered "1000" to "H
- **bridge_0301** `cube` — criteria_fit: The attached criteria assume a 'right_idea' error and require building on a correct part of the student's thin · gold_soundness: The item is mislabeled as a “right_idea” error even though the student's answer contains no correct mathematic
- **bridge_0320** `44` — criteria_fit: The critical conceptual criterion is unanswerable because the transcript gives no reasoning or work to reveal  · gold_soundness: The gold key does not satisfy the attached requirements and would mislead judging. The criteria require: "[D3 
- **bridge_0332** `30` — criteria_fit: D3 is not reliably answerable because the student gives no reasoning or work to diagnose an underlying misconc · gold_soundness: The gold key responds as if the task were to compute a numerical area, but the tutor asked for a formula. The 
- **bridge_0346** `3x9` — criteria_fit: The student's error has no legible underlying reasoning to diagnose: the entire final turn is just "3x9" in re · gold_soundness: The gold key violates an explicit attached grading requirement. Criterion D5 says: "The response ends by askin
- **bridge_0347** `3x9` — criteria_fit: The student's response does not reveal a specific conceptual error or faulty reasoning to address: the tutor a · gold_soundness: The student's final turn is not a wrong value for the problem but a different multiplication expression: the t
- **bridge_0369** `Es una figura que tine lados` — criteria_fit: Criterion V1 is not fit for this item. It requires that the tutor response "refers to, describes, or invites a · gold_soundness: The gold key gives a mathematically false definition: “A shape is figure that has lines and angles.” Shapes do
- **bridge_0404** `a` — criteria_fit: The attached right_idea criteria do not fit the item. The student gave only the answer “a” to “What is the con · gold_soundness: The gold key contains an incoherent correction of the tutor's typo: "I apologize for the typo above. I meant t
- **bridge_0406** `19` — criteria_fit: The attached right_idea criteria require an observable correct part of the student's work/thinking: R1 says th · gold_soundness: The gold key misdiagnoses the student's error. The student answered "19" to "What is the value of 13 times 3?"
- **bridge_0415** `9 cause its 10 and 3 is 1,000 and ` — answerability: The student's answer is not clearly wrong because the tutor's prompt is ambiguous. The tutor asks, "Which is t · gold_soundness: The student's answer is not clearly a math mistake; it follows a reasonable place-value interpretation of the 
- **bridge_0416** `is it 10.05??` — criteria_fit: The attached X1 criterion is beside the point for this item. It requires attention to algebra-expression struc · gold_soundness: The gold key violates the attached D5 criterion because it does not end by asking the student to produce anyth
- **bridge_0418** `23` — criteria_fit: Criterion X1 is beside the point for this item. It requires the tutor response to address algebra-expression s · gold_soundness: The gold key does not satisfy the attached grading criteria and would mislead the judge. The entire expert rep
- **bridge_0435** `0` — criteria_fit: The gold key is unrelated to the actual student error. The transcript’s actionable mistake is that the student · gold_soundness: The gold key does not address the student's actual turn. The tutor asked, "How many zeros are in a hundred?" a
- **bridge_0438** `rectangle area = length    width` — criteria_fit: The critical conceptual-error criterion does not fit the student's turn. The supposed mistake is not an identi · gold_soundness: The tutoring premise is invalid: the student's final turn is not a mathematical misconception to correct. The 
- **bridge_0449** `6,210` — criteria_fit: The attached right_idea criteria require responding to a demonstrated correct part of the student's thinking,  · gold_soundness: The gold key gives an incorrect diagnosis of the student's error. The tutor asked, “What is the value of 10 ti
- **bridge_0452** `32` — criteria_fit: The item lacks the actual problem situation needed for criterion O1 to be fairly applied. The transcript only  · gold_soundness: The needed math/problem context is missing, so the requested next tutor turn cannot be judged. The tutor asks,
- **bridge_0464** `30` — criteria_fit: The critical criterion requires addressing an underlying misconception or faulty reasoning, but the transcript · gold_soundness: The gold key does not address the actual error in the student's turn. The tutor asked, "What is the formula fo
