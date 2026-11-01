"""Frozen LCLD configuration used by both target-training regimes."""
FIXED = dict(steps=200, kappa=40., reinit_round=100, reinit_kappa=0.,
             cont_param='ballv2', ball_sched='se', reinit_dir=True,
             dir_gate=25, se_gate=25, repair_grad='chain', rel_w=0.,
             logits_lr=.48, d_lr=.06, rho_scale=1.,
             dfo_start=None, extra_search=0, stall_restart=False)
CONFIG = {**FIXED, 'logits_lr': .12, 'd_lr': .06}
EXPECTED_KERNEL = 'bb3f5110a24dd0e5bad4999eed96f334b354019af9fc3a99ac482d6e2b7c870d'
