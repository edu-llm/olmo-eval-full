# What to send before we evaluate your checkpoint

Three things, in one message.

1. **The checkpoint's `s3://` prefix**, copied verbatim — the value your training run
   exposed as `EDULLM_CHECKPOINT_DIR`. This is the one item nobody can derive on your
   behalf. The evaluation role can only read under
   `s3://sbsandbox-intern-edullm-outputs/teams/<team>/runs/<run-id>/`, so a checkpoint
   anywhere else is accepted, given a machine, and then denied on its first read.
2. **Which benchmarks to run**, or "whatever is ready" if you have no preference.
3. **Which hardware.** `gpu-1xl4` unless you know you need otherwise: it is what every
   validated run has used, and the platform refuses the Turing shapes for this command.

## You do not need to send a runner

Getting your checkpoint to load and decode is the evaluating agent's first task, and it
does it against your weights directly rather than asking you to demonstrate it. If a
checkpoint turns out to be unreadable that is a finding, and finding it is the job.

Send them anyway if you already have them, because they shorten that first step: the
script you load the model with, as-is rather than cleaned up; roughly 200 tokens of its
real output pasted verbatim; and the library and commit you loaded it with. None of this
blocks anything by being absent.

## If you do send output, send it raw

Four things are visible in real output and in nothing else: whether your tokenizer's pad
and end-of-text ids are the same value, whether the model ever emits an end-of-text token
or runs to the token cap on every prompt, what precision the weights actually load at, and
whether the model repeats its prompt back or loops one clause instead of answering.

The last of those decides whether we can score you on the generative benchmarks at all.
Their graders check a response against the instruction it was given, and an instruction has
to name the thing that satisfies it — so a model that recites its prompt passes items it
never answered, and scores *higher* than one that tried. A description of the output does
not settle any of this; the raw text does, including anything in it that looks broken.
