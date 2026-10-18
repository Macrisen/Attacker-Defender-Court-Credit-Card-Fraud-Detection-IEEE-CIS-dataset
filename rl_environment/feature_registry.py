"""Grounded feature classifications. Scenario assumptions do not rewrite evidence."""
import re

IMMUTABLE = {'transactionid', 'transactiondt', 'isfraud', 'partition'}


def feature_group(name):
    if name in IMMUTABLE:
        return 'metadata'
    if name == 'transactionamt':
        return 'TransactionAmt'
    if name == 'productcd':
        return 'ProductCD'
    for prefix in ('card', 'addr', 'dist', 'id_'):
        if name.startswith(prefix):
            return prefix.rstrip('_')
    if 'emaildomain' in name:
        return 'email_domain'
    if re.fullmatch(r'[cdmv]\d+', name):
        return name[0].upper()
    return 'other_identity_or_transaction'


def audit_schema(schema, version='cleaned'):
    rows = []
    for field in schema:
        meta = field.name in IMMUTABLE
        amount = field.name == 'transactionamt'
        rows.append({'feature': field.name, 'dataset_version': version, 'data_type': str(field.type),
            'feature_group': feature_group(field.name), 'classification': 'IMMUTABLE' if meta else 'UNKNOWN',
            'perturbation_allowed': False, 'candidate_attack_control': 'amount_under_scenario_B_only' if amount else '',
            'dependency_status': 'NOT_EDITABLE' if meta else 'UNKNOWN',
            'label_preservation_assumption': 'hypothetical intent invariant; NOT observed' if amount else 'none established',
            'evidence_level': 'schema_and_project_contract' if meta else 'schema_only; no verified causal control',
            'validation_rules': 'exact preservation' if meta else 'reject independent mutation in Scenario A',
            'justification': 'ID/time/label protected' if meta else 'No defensible control or recomputation rule supplied',
            'limitations': 'No predictor semantics inferred from type, name, correlation, or model importance'})
    return rows


class FeatureRegistry:
    def __init__(self, entries):
        self.entries = {e['feature']: dict(e) for e in entries}
        if len(self.entries) != len(entries):
            raise ValueError('Duplicate feature registry keys')

    def authorize(self, controls, scenario):
        for control in controls:
            name = control['feature']
            if name in IMMUTABLE:
                raise ValueError('Immutable control: '+name)
            if name not in self.entries:
                raise ValueError('Unregistered control: '+name)
            classification = self.entries[name]['classification']
            if classification == 'MUTABLE':
                if not self.entries[name].get('evidence_level'):
                    raise ValueError('Mutable control requires evidence')
            elif scenario != 'B' or name != 'transactionamt' or classification != 'UNKNOWN':
                raise ValueError('Unknown/immutable feature cannot be authorized: '+name)
        return True

    @classmethod
    def synthetic(cls):
        return cls([{'feature': n, 'classification': c, 'evidence_level': 'controlled generator contract'}
                    for n, c in [('quantity', 'MUTABLE'), ('unit_price', 'IMMUTABLE'), ('total', 'DEPENDENT')]])
