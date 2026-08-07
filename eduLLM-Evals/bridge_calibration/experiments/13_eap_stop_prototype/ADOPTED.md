# ADOPTED

The EAP-posterior-SD stop prototyped here (Phase A) is now **of-record** for Bridge at the
locked operating point **min_scenarios 12 / SE-ability target 0.12** (Phase B adoption).

- Stop rule: online normal-approx SE -> **EAP posterior SD** (fine 321-node grid over +/-8).
- SE target: historical **0.15 -> 0.12** (tail control of SE_total; aligns with WildBench/BiGGen).
- Deployed of-record artifacts re-run under the EAP stop: exp-04, exp-05, exp-09, exp-10,
  exp-11, and the deployed-SE artifacts in exp-07 (see ../07_parameter_uncertainty/).
- Operating-point grid promoted to ../06b_operating_point/ (from ../13_eap_stop_grid/).
- Prior online-SE @0.15 of-record archived at ../archive_onlineSE/.

This prototype directory (and 13_eap_stop_grid/) are retained as Phase-A provenance.
