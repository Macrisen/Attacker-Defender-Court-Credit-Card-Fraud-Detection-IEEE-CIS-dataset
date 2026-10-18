"""Declarative, evidence-gated dependency recomputation. No inferred causal rules."""
import math
from .feature_registry import IMMUTABLE


class DependencyValidator:
    def __init__(self, status='UNKNOWN', rules=()):
        self.status, self.rules = status, list(rules)
        for rule in self.rules:
            if rule['output'] in IMMUTABLE or rule['operation'] != 'multiply':
                raise ValueError('Unsupported or immutable dependency output')
            if len(rule['inputs']) != 2:
                raise ValueError('multiply requires exactly two inputs')

    def recompute(self, transaction):
        for rule in self.rules:
            a, b = (transaction[n] for n in rule['inputs'])
            if a is None or b is None:
                raise ValueError('Missing controlling value; no implicit imputation')
            transaction[rule['output']] = a*b

    def validate(self, transaction):
        if self.status not in ('VERIFIED_FIXTURE', 'VERIFIED'):
            return 'UNKNOWN', 'Dependency graph unverified; keeping other predictors fixed is not established'
        for rule in self.rules:
            a, b = (transaction[n] for n in rule['inputs'])
            actual = transaction[rule['output']]
            if actual is None or not math.isfinite(float(actual)) or not math.isclose(float(actual), float(a*b), rel_tol=1e-12, abs_tol=1e-12):
                return 'INVALID', 'Dependent value mismatch: '+rule['output']
        return 'VALID', 'All registered dependencies satisfied'
