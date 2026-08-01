import os

import pandas as pd

pd.set_option("display.width", 250)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "_scratch_atlas")
df = pd.read_csv(os.path.join(OUT, "candidates_flagged.csv"))

print("Qwen2.5 entries in pools:")
print(df[df.model.str.contains("Qwen2.5", case=False, na=False, regex=False)][
    ["model", "params_b", "family", "n_flags", "reputable_org"]].head(30).to_string(index=False))

print()
print("zephyr from HuggingFaceH4 / mirror orgs present?")
for probe in ["HuggingFaceH4/", "unsloth/", "bartowski/", "TheBloke/", "prunaai/",
              "mradermacher/", "lmstudio-community/", "ModelCloud/", "neuralmagic/",
              "RedHatAI/", "second-state/", "nm-testing/"]:
    hit = df[df.model.str.lower().str.startswith(probe.lower(), na=False)]
    print(f"  {probe:<22} n={len(hit):<4} {', '.join(hit['model'].head(6))}")

print()
print("gemma family assignment check:")
print(df[df.model.str.contains("^google/gemma", case=False, na=False)][
    ["model", "params_b", "family"]].to_string(index=False))

print()
print("llama3x-without-separator ids:")
print(df[df.model.str.contains("llama3[12]", case=False, na=False)][
    ["model", "family"]].head(15).to_string(index=False))

print()
print("ontocord entries:")
print(df[df.org.str.lower() == "ontocord"][["model", "params_b", "n_flags"]].to_string(index=False))
