"""
How can we be sure the hand-set numbers do not decide the result?  -> a parameter-recovery test
================================================================================================
1. Make a "TRUE" network whose tables differ from our hand-set guesses (random, sizeable changes).
   In reality we never know the truth; here we do, so we can measure how close learning gets.
2. Sample questions from the truth, hide the causes (keep a weak, 5%-noisy label on a share of them,
   as the datasets will provide), hide some LLM-tier signals.
3. Learn with different starting points (hand-set / random), prior strengths alpha, and weak-label
   rates, and compare against the truth on fresh test questions:
     table error      mean |learned - true| over all table entries
     cause accuracy   does the posterior P(cause = High) > 0.5 agree with the true cause?
                      (do the hidden nodes keep their MEANING?)
     outcome log-loss how well P(Correct / Abstain / Wrong) is predicted (truth = best possible)
     agreement        same action as a router that knows the true numbers
     regret           utility lost per question vs that router (true utilities)
If the results with a random start match those with the hand-set start, the hand-set numbers are not
what produces the result.

    python recovery_test.py              # ~15 minutes on a laptop
    python recovery_test.py --quick      # smaller
"""

import argparse
import copy
import time

import numpy as np

import ber_v3 as B
import learn_v3 as L


def perturbed_truth(prior: B.Params, seed, concentration=8.0, effect_shift=0.15):
    """A plausible 'reality' that is clearly different from our guesses."""
    rng = np.random.default_rng(seed)
    T = L.random_init(prior, rng, concentration=concentration)
    eff = copy.deepcopy(prior.effects)
    for path, v in L._flatten(eff):
        L._set(eff, path, float(np.clip(v + rng.uniform(-effect_shift, effect_shift), 0.02, 0.98)))
    return B.Params(T.cause, T.symptom, T.outcome, eff)


def effect_error(P, T):
    a = np.array([v for _, v in L._flatten(P.effects)])
    b = np.array([v for _, v in L._flatten(T.effects)])
    return float(np.abs(a - b).mean())


def evaluate(P, truth, test_rows, d_test, n_dec=400):
    r_true, r = B.Router(truth), B.Router(P)
    # cause accuracy on the test set, from symptoms only
    q, _ = L._normalise(L.log_joint(r, d_test, use_labels=False, use_outcome=False))
    z_true = B.Z[[row["z"] for row in test_rows]]
    acc = float(((q @ B.Z) > 0.5).astype(int).__eq__(z_true).mean())
    ll = L.outcome_logloss(P, d_test)
    agree, regret = 0, 0.0
    for row in test_rows[:n_dec]:
        ev, ctx = row["symptoms"], row["context"]
        typed = ev.get("SubjectTypeMatch") in ("Match", "Mismatch")
        choice = r.decide(ev, ctx)["action"]
        qt = r_true.posterior(ev)
        eu = {a: r_true.expected_utility(a, qt, {**B.DEFAULT_CONTEXT, **ctx}, typed) for a in B.ACTIONS}
        best = max(eu, key=eu.get)
        agree += choice == best
        regret += eu[best] - eu[choice]
    return {"table_err": L.table_distance(P, truth), "cause_acc": acc, "logloss": ll,
            "agree": agree / n_dec, "regret": regret / n_dec, "effect_err": effect_error(P, truth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    n_train, n_test = (3000, 1000) if args.quick else (8000, 2000)
    prior = B.handset_params()
    truth = perturbed_truth(prior, args.seed)
    print(f"truth differs from the hand-set guesses by {L.table_distance(prior, truth):.3f} "
          f"(mean |difference| per table entry), action effects by {effect_error(prior, truth):.3f}")
    missing = {2: 0.2, 3: 0.4}
    test_rows = B.sample(truth, n_test, seed=args.seed + 100, missing=missing)
    d_test = L.Data(test_rows)

    head = f"{'setting':<46}{'table err':>10}{'cause acc':>10}{'log-loss':>10}{'agree':>8}{'regret':>8}{'effects':>9}"
    print("\n" + head + "\n" + "-" * len(head))

    def show(name, m):
        print(f"{name:<46}{m['table_err']:>10.3f}{m['cause_acc']:>10.1%}{m['logloss']:>10.4f}"
              f"{m['agree']:>8.1%}{m['regret']:>8.3f}{m['effect_err']:>9.3f}")

    show("TRUE numbers (best possible)", evaluate(truth, truth, test_rows, d_test))
    show("hand-set guesses, no learning", evaluate(prior, truth, test_rows, d_test))

    rng = np.random.default_rng(args.seed)
    rand_start = L.random_init(prior, rng)
    for label_rate in (0.0, 0.3):
        rows = B.sample(truth, n_train, seed=args.seed + int(label_rate * 100), missing=missing,
                        label_rate=label_rate, action_rate=0.3)
        d = L.Data(rows)
        print(f"-- weak cause labels on {label_rate:.0%} of rows ({n_train} training questions) --")
        for alpha in (0.0, 5.0, 50.0):
            for start_name, start in (("hand-set start", prior), ("random start", rand_start)):
                t = time.time()
                P, _, it = L.em(d, prior, alpha, init=start)
                P = L.fit_effects(P, d, prior.effects, verbose=False)
                show(f"   alpha {alpha:<5} {start_name:<16} ({it:>3} it)", evaluate(P, truth, test_rows, d_test))
        a_cv, _ = L.select_alpha(d, prior, alphas=(0.0, 1.0, 5.0, 20.0, 50.0), k=3, verbose=False)
        P, info = L.fit_with_restarts(d, prior, a_cv, n_random=3, seed=args.seed)
        P = L.fit_effects(P, d, prior.effects, verbose=False)
        show(f"   PIPELINE: alpha by CV = {a_cv} + 4 restarts", evaluate(P, truth, test_rows, d_test))
        print(f"      distance between restarts: {info['max_distance_between_starts']}")

    print("\nHow much data? (weak labels on 30%, alpha by CV, hand-set start)")
    for n in (1000, 3000, 8000):
        rows = B.sample(truth, n, seed=args.seed + n, missing=missing, label_rate=0.3, action_rate=0.3)
        d = L.Data(rows)
        a_cv, _ = L.select_alpha(d, prior, alphas=(0.0, 1.0, 5.0, 20.0, 50.0), k=3, verbose=False)
        P, _, _ = L.em(d, prior, a_cv)
        P = L.fit_effects(P, d, prior.effects, verbose=False)
        show(f"   n = {n:<6} (alpha {a_cv})", evaluate(P, truth, test_rows, d_test))


if __name__ == "__main__":
    main()
