"""Executed numerical checks shared by tests and the readiness CSV report."""
import math
from .court_credit_assignment import DelayedCreditLedger, time_discount, dqn_target


def event(action=0, time=172800, source='court', bonus=10):
    return {'action_id': action, 'transaction_id': action+10, 'request_time': 0,
            'completion_time': time, 'source': source, 'reward': bonus}


def credit_checks():
    rows = []
    def check(name, actual, expected):
        passed = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12) if isinstance(expected, float) else actual == expected
        rows.append({'test_name': name, 'expected': expected, 'actual': actual, 'status': 'PASS' if passed else 'FAIL'})
    ledger = DelayedCreditLedger(400000, .5)
    for i in range(101):
        ledger.register(i, i+10, 0 if i == 0 else i*1728, [i], int(i == 0), i == 0, 'rl_development')
        ledger.attach_next(i, [i+1], (i+1)*1728)
    check('origin_not_ready_before_verification', any(r['court_action_id'] == 0 for r in ledger.ready(1000)), False)
    ledger.complete(event(), 172800)
    origin = ledger.records()[0]
    check('bonus_at_step_100_credited_to_origin_zero', origin['final_attributed_reward'], 1.5)
    check('unrelated_step_100_reward_zero', ledger.records()[100]['final_attributed_reward'], 0.)
    check('cost_charged_once', origin['cost'], 1.)
    ready = ledger.ready(174528)
    origin_replay = next(r for r in ready if r['court_action_id'] == 0)
    check('time_discount_two_days', time_discount(172800, .5), .25)
    check('td_target_uses_time_to_original_next_state', dqn_target(origin_replay, 4.), 1.5+(.5**.02)*4.)
    check('replay_record_emitted_once', len(ledger.ready(174528)), 0)
    try:
        ledger.complete(event(), 172800)
        duplicate_rejected = False
    except ValueError:
        duplicate_rejected = True
    check('duplicate_completion_rejected', duplicate_rejected, True)
    for name, partition, end, trunc, censor in [('boundary', 'rl_development', 100000, False, False),
          ('holdout', 'final_holdout', 400000, False, False), ('truncation', 'rl_development', 400000, True, False),
          ('censored', 'rl_development', 400000, False, True)]:
        l = DelayedCreditLedger(end, .5)
        l.register(0, 10, 0, [0], 1, True, partition)
        l.attach_next(0, [1], 10, truncated=trunc)
        if censor:
            l.censor_pending()
        l.complete(event(), 172800)
        check(name+'_not_training_eligible', len(l.ready(172800)), 0)
    l = DelayedCreditLedger(400000)
    l.register(0, 10, 0, [0], 0, False, 'rl_development')
    l.attach_next(0, [1], 1)
    try:
        l.complete(event(), 172800)
        skip_rejected = False
    except ValueError:
        skip_rejected = True
    check('skip_cannot_receive_verification_bonus', skip_rejected, True)
    check('audit_event_not_credited_to_court', l.complete(event(source='audit'), 172800), False)
    gamma = .9
    actual = (-1+gamma**2*3)+gamma*(-1+gamma**2*10)
    expected = -1-gamma+gamma**2*3+gamma**3*10
    check('discounted_return_equivalence', actual, expected)
    return rows
