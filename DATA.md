# Data inputs and access

The study uses previously recorded OLX Jobs interactions. Obtain the corresponding source data through an authorized source and comply with its actual access and reuse conditions. This repository does not grant rights to the dataset or assert that raw data are openly redistributable.

The local preprocessing input has four columns: `user`, `item`, `event`, and `timestamp`. Events include `click`, `bookmark`, `chat_click`, `contact_chat`, `contact_phone_click_1`, `contact_phone_click_2`, `contact_phone_click_3`, and `contact_partner_click`. Timestamps are the released time-shifted Unix values; UTC conversion does not recover the real calendar dates.

Preprocessing removes exact duplicate events and creates deterministic integer mappings. Sequence construction adds user-job sequence IDs, event positions, preceding events and transition delays. The main experiment expects:

- `data/olx/04_sequences/ui_sequences.parquet`: events with `user_idx`, `item_idx`, `event`, `timestamp`, `sequence_id`, `position`, `previous_event`, `delta_seconds` and temporal-transition fields.
- `data/olx/09_graphs/recommendation_pilot/recommendation_eval_targets.parquet`: source attempts paired with eligible subsequent alternative-job contacts. This is an intermediate input, not a public result table.
- Generated files below `experiments/olx/`: private feature tables, prediction files, candidate arrays, input manifests and per-user ranking outputs. Generate them locally; do not commit them.

The conditional target is a later qualifying contact on a different previously unseen job within 24 hours. Recorded contact actions are behavioral proxies, not verified applications, interviews or hires. Unobserved comparison jobs are not verified dislikes.

The cutoff-observed catalog is an availability proxy. Listing publication/expiry fields and independent subsequent data are not supplied here. Taobao and Tmall are not part of this result release.

## REES46 Cosmetics Shop

The secondary experiment uses [eCommerce Events History in Cosmetics Shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop), distributed by Michael Kechinov with attribution to REES46 Marketing Platform. The source covers October 2019–February 2020; this study uses October–December only. Obtain the source under its actual access/reuse terms. Research-use availability does not imply permission to redistribute raw events.

Supply `2019-Oct.csv.zip`, `2019-Nov.csv.zip` and `2019-Dec.csv.zip` locally. Each ZIP must contain its correspondingly named CSV. Required columns are `event_time`, `event_type`, `product_id` and `user_id`. The code uses recorded views, carts, purchases and removals, collapsing identical timestamp/user/product/event keys. It does not interpret cart removal or absent purchase as verified dislike.

Keep source archives and derived user-level files outside tracked paths; defaults are the ignored `local_data/cosmetics/` and `local_work/cosmetics/`. No credentials, cookies, API keys, user identifiers, paired ranks or candidate arrays are published. Source archive hashes are in `results/cosmetics/data_provenance.json`.

Only aggregate, non-user-level research outputs are included. Dataset access and institutional requirements must be resolved with the data provider and the researcher’s institution.
