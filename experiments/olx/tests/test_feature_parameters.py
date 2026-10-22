import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rebuild_features import fit_parameters, context_sql


class ParameterTests(unittest.TestCase):
    def test_future_transitions_do_not_change_fit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'events.parquet'
            data=pd.DataFrame({'previous_event':['a','a','b','a'],
                'event':['b','c','a','b'],'delta_seconds':[1,2,3,999],
                'timestamp':[10,20,30,100],'valid_temporal_transition':[True]*4})
            data.to_parquet(path)
            con=duckdb.connect()
            before=fit_parameters(con,path.as_posix(),100)
            data.loc[3,['delta_seconds','event']]=[99999,'c']
            data.to_parquet(path)
            after=fit_parameters(con,path.as_posix(),100)
            pd.testing.assert_frame_equal(before,after)
            self.assertEqual(before.transition_count.sum(),3)
            con.close()

    def test_context_excludes_equal_time_and_other_pairs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path=Path(directory)/'events.parquet'
            pd.DataFrame({'sequence_id':[1,1,1,1,2],'position':[0,1,2,3,0],
                'timestamp':[10,20,20,30,5],
                'event':['click','bookmark','chat_click','contact_chat','click']}).to_parquet(path)
            con=duckdb.connect()
            result=con.execute(context_sql(path.as_posix())).fetchdf().set_index(['sequence_id','position'])
            self.assertEqual(result.loc[(1,1),'context_prior_event_count'],1)
            self.assertEqual(result.loc[(1,2),'context_prior_event_count'],1)
            self.assertEqual(result.loc[(1,3),'context_prior_event_count'],3)
            self.assertEqual(result.loc[(1,3),'previous_time'],20)
            self.assertEqual(result.loc[(2,0),'context_prior_event_count'],0)
            con.close()


if __name__=='__main__':
    unittest.main()
