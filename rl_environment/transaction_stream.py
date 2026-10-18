"""Internal chronological records; labels never belong in agent observations."""
from dataclasses import dataclass
import copy
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

METADATA = {'transactionid', 'transactiondt', 'isfraud'}


@dataclass(frozen=True)
class InternalTransaction:
    transaction_id: int
    transaction_time: int
    features: dict
    hidden_label: int


class TransactionStream:
    def __init__(self, parquet, boundaries, feature_names, limit=10000):
        self.path = Path(parquet)
        if not self.path.exists():
            raise FileNotFoundError(f'Dataset NOT AVAILABLE: {self.path}')
        split = json.loads(Path(boundaries).read_text())
        self.cutoff = split['model_training_cutoff']
        if self.cutoff >= split['prior_holdout_cutoff']:
            raise ValueError('Invalid saved model-training / holdout boundaries')
        pf = pq.ParquetFile(self.path)
        names = pf.schema_arrow.names
        if set(names) != set(feature_names) | METADATA:
            pf.close()
            raise ValueError('Labeled Parquet schema mismatch: Kaggle test cannot supply fraud rewards')
        if METADATA & set(feature_names):
            pf.close()
            raise ValueError('Metadata must not be predictors')
        self.schema_names = names
        self.source_rows = pf.metadata.num_rows
        self.null_counts = {name: sum(pf.metadata.row_group(i).column(j).statistics.null_count
                           for i in range(pf.num_row_groups)) for j, name in enumerate(names)}
        meta = pf.read(columns=['transactionid', 'transactiondt'])
        pf.close()
        if any(col.null_count for col in meta.columns):
            raise ValueError('NULL transaction ID or time')
        if len(pc.unique(meta['transactionid'])) != len(meta):
            raise ValueError('Duplicate TransactionID')
        eligible = meta.filter(pc.less_equal(meta['transactiondt'], self.cutoff))
        if len(eligible) != split['train_end']:
            raise ValueError('Model-training partition count differs from baseline')
        eligible = eligible.sort_by([('transactiondt', 'ascending'), ('transactionid', 'ascending')])
        if limit is not None:
            if limit <= 0:
                raise ValueError('limit must be positive')
            eligible = eligible.slice(0, limit)
        filt = (ds.field('transactiondt') <= self.cutoff) & ds.field('transactionid').isin(eligible['transactionid'])
        self._table = ds.dataset(self.path, format='parquet').to_table(filter=filt).sort_by(
            [('transactiondt', 'ascending'), ('transactionid', 'ascending')])
        if len(self._table) != len(eligible) or not self._table['transactionid'].equals(eligible['transactionid']):
            raise ValueError('Selected rows differ from chronological metadata')
        labels = self._table['isfraud'].to_pylist()
        if any(v not in (0, 1) for v in labels):
            raise ValueError('Invalid or missing training label')
        self._records = [InternalTransaction(int(r['transactionid']), int(r['transactiondt']),
                         {n: r[n] for n in feature_names}, int(r['isfraud']))
                         for r in self._table.to_pylist()]
        self.reset()

    @classmethod
    def synthetic(cls, records):
        obj = cls.__new__(cls)
        obj._records = sorted(copy.deepcopy(records), key=lambda r: (r.transaction_time, r.transaction_id))
        if len({r.transaction_id for r in records}) != len(records):
            raise ValueError('Duplicate TransactionID')
        if any(r.hidden_label not in (0, 1) or METADATA & set(r.features) for r in records):
            raise ValueError('Invalid internal fixture')
        obj.path = None
        obj.reset()
        return obj

    def reset(self):
        self.position = 0

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_transaction()

    def next_transaction(self):
        if self.position >= len(self):
            raise StopIteration
        record = copy.deepcopy(self._records[self.position])
        self.position += 1
        return record

    def batch(self, size):
        if size <= 0:
            raise ValueError('batch size must be positive')
        return [self.next_transaction() for _ in range(min(size, len(self)-self.position))]

    def internal_records(self):
        """Controller-only access. Never pass this collection to an agent."""
        return copy.deepcopy(self._records)
