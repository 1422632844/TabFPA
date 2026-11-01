"""LOCAL WRAPPER / AUDIT: LCLD adapters for the released Gumbel loop.

Nonlinear equalities are enforced by conditional continuous completion, not
discarded. The existing attack loop, logits loss and query accounting are reused.
"""
from types import SimpleNamespace
import numpy as np
from core.feasible_integer_moves import JointIntegerMoves
from core.conditional_completion import ConditionalCompletion


class LcldMoves(JointIntegerMoves):
    def __init__(self, meta, scaler, cons, eps=.495):
        names = meta.feature.tolist()
        self.j = {name: i for i, name in enumerate(names)}
        frozen = ["term", "int_rate", "pub_rec", "pub_rec_bankruptcies",
                  "month_since_earliest_cr_line",
                  "ratio_pub_rec_month_since_earliest_cr_line",
                  "ratio_pub_rec_bankruptcies_month_since_earliest_cr_line",
                  "ratio_pub_rec_bankruptcies_pub_rec"]
        assert not meta.iloc[[self.j[n] for n in frozen]].mutable.any()
        assert len(cons.relation_constraints) == 9
        # These are the native open<=total and bankruptcies<=public-records
        # inequalities. The other relations are either fixed by immutable
        # inputs or satisfied by the continuous witness below/completion.
        affine = SimpleNamespace(relation_constraints=cons.relation_constraints[1:3])
        super().__init__(meta, scaler, affine, eps)
        assert set(meta.iloc[self.ints].feature) == {"emp_length", "open_acc", "total_acc", "mort_acc"}
        self.op = self.j["open_acc"]
        self.total = self.j["total_acc"]
        self.ratio = self.j["ratio_open_acc_total_acc"]

    def candidates(self, clean, current):
        values, mask = super().candidates(clean, current)
        # Include the derived real ratio in every discrete move's full cost
        # BEFORE Gumbel sampling. A ratio-only continuous witness is explicit.
        initial_cost = (((current-clean)/self.scale)**2).sum(1)
        for direction, support in enumerate(self.support):
            step = values[:, direction]
            op = current[:, self.op, None] + step*self.D[direction, self.op]
            total = current[:, self.total, None] + step*self.D[direction, self.total]
            ratio = op/total
            changed = current[:, support, None] + self.D[direction, support][None, :, None]*step[:, None, :]
            before = ((current[:, support]-clean[:, support])/self.scale[support])**2
            after = ((changed-clean[:, support, None])/self.scale[support][None, :, None])**2
            cost = initial_cost[:, None] - before.sum(1)[:, None] + after.sum(1)
            cost += ((ratio-clean[:, self.ratio, None])/self.scale[self.ratio])**2
            cost -= ((current[:, self.ratio]-clean[:, self.ratio])/self.scale[self.ratio])[:, None]**2
            mask[:, direction] &= (total > 0) & (ratio >= self.lo[self.ratio]) & (ratio <= self.hi[self.ratio]) & (cost <= self.eps**2+1e-13)
        return values, mask

    def sample(self, current, values, mask, theta, tau):
        raw, straight, counts = super().sample(current, values, mask, theta, tau)
        raw[:, self.ratio] = raw[:, self.op]/raw[:, self.total]
        return raw, straight, counts


class LcldCompletion(ConditionalCompletion):
    def __init__(self, dataset, meta, scaler, judge, eps=.495):
        super().__init__(dataset, meta, scaler, judge, eps)
        self.kind = "LCLD_analytic_anchor_and_local_SLSQP_completion"

    def _lcld(self, x0, p):
        # With all independent real inputs at clean values, only open/total
        # changes. Keeping the other real fields exactly clean is a global
        # minimum-distance continuous witness for these fixed integer choices.
        j = self.j
        anchor = (p[:, self.free] == x0[:, self.free]).all(1)
        out = p.copy()
        rows = np.flatnonzero(anchor)
        out[np.ix_(rows, self.real)] = x0[np.ix_(rows, self.real)]
        out[:, j["ratio_open_acc_total_acc"]] = p[:, j["open_acc"]]/p[:, j["total_acc"]]
        valid = np.ones(len(p), dtype=bool)
        other = np.flatnonzero(~anchor)
        if len(other):
            out[other], valid[other] = super()._lcld(x0[other], p[other])
        return out, valid
