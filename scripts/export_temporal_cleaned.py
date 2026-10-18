"""Deterministic column projection; no fitting or row filtering during export."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pyarrow.parquet as pq
from preprocessing.temporal_feature_audit import sha
from rl_environment.temporal_protocol import verify_alignment

def main():
    manifest=json.loads((ROOT/'reports/phase3_6/selection_manifest.json').read_text())
    results=[]
    for kind in ('train','test'):
        source=ROOT/f'data/processed/{kind}_original.parquet'
        dest=ROOT/f'data/processed/{kind}_{manifest["version"]}.parquet'
        if dest.exists():raise FileExistsError(dest)
        names=manifest['selected_predictors']+['transactionid','transactiondt']+(['isfraud'] if kind=='train' else [])
        pf=pq.ParquetFile(source);temporary=dest.with_suffix('.tmp')
        try:
            with pq.ParquetWriter(temporary,pf.schema_arrow.empty_table().select(names).schema,compression='zstd') as writer:
                for i in range(pf.num_row_groups):writer.write_table(pf.read_row_group(i,columns=names))
            saved=pq.ParquetFile(temporary)
            assert saved.metadata.num_rows==pf.metadata.num_rows
            # Every projected value and NULL checked, including metadata.
            for i in range(pf.num_row_groups):assert pf.read_row_group(i,columns=names).equals(saved.read_row_group(i))
            verify_alignment(source,temporary)
            saved.close();temporary.rename(dest)
        finally:
            pf.close();temporary.unlink(missing_ok=True)
        results.append(dict(path=str(dest),sha256=sha(dest),rows=pq.read_metadata(dest).num_rows,columns=len(names),values_verified=True,transform='column projection only'))
    (ROOT/'reports/phase3_6/cleaned_projection_validation.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
