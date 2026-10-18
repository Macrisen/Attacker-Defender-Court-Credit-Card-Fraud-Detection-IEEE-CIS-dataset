"""Run all three phases and save actual per-test outcomes."""
import sys,unittest,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import pandas as pd
rows=[]
class Results(unittest.TextTestResult):
    def startTest(self,test):self.started=time.monotonic();super().startTest(test)
    def record(self,test,status,detail=''):
        rows.append(dict(test=test.id(),status=status,seconds=time.monotonic()-self.started,detail=detail))
    def addSuccess(self,test):super().addSuccess(test);self.record(test,'PASS')
    def addFailure(self,test,err):super().addFailure(test,err);self.record(test,'FAIL',self._exc_info_to_string(err,test))
    def addError(self,test,err):super().addError(test,err);self.record(test,'ERROR',self._exc_info_to_string(err,test))
    def addSkip(self,test,reason):super().addSkip(test,reason);self.record(test,'SKIP',reason)
    def addSubTest(self,test,subtest,err):
        super().addSubTest(test,subtest,err)
        if err:self.record(subtest,'FAIL',self._exc_info_to_string(err,test))
if __name__=='__main__':
    suite=unittest.TestSuite()
    for folder in ('rl_environment','phase3_5','phase3_6'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'/folder)))
    result=unittest.TextTestRunner(verbosity=2,resultclass=Results).run(suite)
    pd.DataFrame(rows).to_csv(ROOT/'reports/phase3_6/phase3_6_test_results.csv',index=False)
    sys.exit(not result.wasSuccessful())
