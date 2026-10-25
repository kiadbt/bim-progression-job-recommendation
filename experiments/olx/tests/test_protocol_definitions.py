"""Regression checks for component horizons and target-filter ordering."""
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from generate_bim_features_successful_contact_olx import pathway_parameters


class ProtocolDefinitionTests(unittest.TestCase):
    def test_phone_uses_maximum_component_quantiles(self):
        rows = [
            ('chat_click', 'contact_chat', 5, .5, 2., 1, 2, 3, 4),
            ('contact_phone_click_1', 'contact_phone_click_2', 90, .3, 2., 1, 2, 12, 30),
            ('contact_phone_click_1', 'contact_phone_click_3', 10, .1, 3., 1, 2, 20, 25),
        ]
        columns = ['source_event', 'target_event', 'transition_count', 'conditional_probability',
                   'lift', 'median_seconds', 'q75_seconds', 'q90_seconds', 'q95_seconds']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'parameters.csv'
            pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
            phone = pathway_parameters(path, path)['contact_phone_click_1']
        self.assertEqual((phone['q90'], phone['q95']), (20, 30))

    def selected_cases(self):
        con = duckdb.connect()
        self.addCleanup(con.close)
        con.execute('CREATE TABLE events(user_idx BIGINT, item_idx BIGINT, timestamp BIGINT, event VARCHAR)')
        rows = [(u, 10, 1, 'click') for u in range(1, 5)]
        rows += [(u, 10, 100, 'chat_click') for u in range(1, 5)]
        rows += [(99, item, 2, 'click') for item in [20, 30, 50, 60, 70, 80]]
        rows += [(1, 20, 90, 'click'), (1, 20, 110, 'contact_chat'), (1, 30, 120, 'contact_chat'),
                 (2, 40, 110, 'contact_chat'), (2, 50, 120, 'contact_chat'),
                 (3, 60, 110, 'contact_chat'), (4, 80, 110, 'contact_chat'), (4, 70, 110, 'contact_chat')]
        con.executemany('INSERT INTO events VALUES (?,?,?,?)', rows)
        con.execute('CREATE TABLE features(attempt_id BIGINT, split VARCHAR, user_idx BIGINT, item_idx BIGINT, source_timestamp BIGINT, source_event VARCHAR)')
        con.executemany('INSERT INTO features VALUES (?,?,?,?,?,?)',
                        [(u, 'validation', u, 10, 100, 'chat_click') for u in range(1, 5)])
        source = (ROOT / 'src/build_recommendation_pilot_olx.py').read_text(encoding='utf-8')
        start = source.index('WITH attempts AS (')
        query = source[start:source.index(") TO '{target_out}'", start)]
        query = query.replace("read_parquet('{feature}')", 'features').replace("read_parquet('{sequence}')", 'events')
        query = query.format(TARGET_HORIZON_SECONDS=86400,
                             SUCCESS_EVENTS="('contact_chat','contact_phone_click_2','contact_phone_click_3')")
        con.execute('CREATE TEMP TABLE targets AS ' + query)
        first = dict(con.execute('SELECT attempt_id,target_item_idx FROM targets').fetchall())
        con.execute('CREATE TEMP TABLE repaired_features AS SELECT attempt_id,split FROM features')
        con.execute('CREATE TEMP TABLE pairs AS SELECT user_idx,item_idx,min(timestamp) first_ts FROM events GROUP BY user_idx,item_idx')
        con.execute('CREATE TEMP TABLE items AS SELECT item_idx,min(first_ts) first_ts FROM pairs GROUP BY item_idx')
        con.execute('CREATE TEMP TABLE users AS SELECT user_idx,min(first_ts) first_ts FROM pairs GROUP BY user_idx')
        source = (ROOT / 'experiments/olx/src/materialize_evaluation_contract.py').read_text(encoding='utf-8')
        start = source.index('WITH eligible AS (')
        query = source[start:source.index('""").fetchdf()', start)]
        retained = con.execute(query.format(TRAIN_END=50, VALIDATION_END=100000, end=200000)).fetchdf()
        return first, dict(zip(retained.attempt_id, retained.target_item_idx))

    def test_no_fallback_after_seen_or_out_of_catalog_first_contact(self):
        first, retained = self.selected_cases()
        self.assertEqual(first[1], 20)
        self.assertEqual(first[2], 40)
        self.assertNotIn(1, retained)
        self.assertNotIn(2, retained)

    def test_eligible_contact_and_target_ties(self):
        _, retained = self.selected_cases()
        self.assertEqual(retained, {3: 60, 4: 70})


if __name__ == '__main__':
    unittest.main()
