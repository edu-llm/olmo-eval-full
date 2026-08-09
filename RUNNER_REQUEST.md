# What to send before we evaluate your checkpoint

Four things, in one message. None of them can be guessed on your behalf.

1. **The checkpoint's `s3://` prefix**, copied verbatim — the value your training run
   exposed as `EDULLM_CHECKPOINT_DIR`. The evaluation role can only read under
   `s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/<run-id>/`, so a checkpoint
   anywhere else is accepted, given a machine, and then denied on its first read.
2. **The script you load and run the model with**, as-is rather than cleaned up. A runner
   file, a notebook cell or three lines in a REPL are all fine.
3. **Roughly 200 tokens of that script's real output, pasted verbatim.** Not a summary,
   not "it generates fine", not the first line — the raw text, including anything in it
   that looks broken.
4. **The library and commit you loaded it with**: package and version, or repository and
   sha. Releases genuinely disagree about which checkpoint configs they can parse, so a
   checkpoint that loads for you can be unreadable to a different build.

## Why item 3 has to be raw

Four things are visible in real output and in nothing else: whether your tokenizer's pad
and end-of-text ids are the same value, whether the model ever emits an end-of-text token
or runs to the token cap on every prompt, what precision the weights actually load at, and
whether the model repeats its prompt back or loops one clause instead of answering.

The last of those decides whether we can score you on the generative benchmarks at all.
Their graders check a response against the instruction it was given, and an instruction has
to name the thing that satisfies it — so a model that recites its prompt passes items it
never answered, and scores *higher* than one that tried. Two hundred tokens of your own
decode settle that before any GPU time is spent. A description of the output does not.
