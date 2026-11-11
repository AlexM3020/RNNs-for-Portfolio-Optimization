# Data snapshots

The files in [snapshots](snapshots/) are copied unchanged from the results repository. They are historical inputs, not fresh downloads.

| File | Purpose |
|---|---|
| [2010–2024 prices](snapshots/etf_data_2010_2024.csv) | Supported example input: 3,773 rows, 2010-01-04 through 2024-12-30 |
| [2010–2022-named snapshot](snapshots/etf_data_2010_2022.csv) | Earlier filename/version retained for provenance; filename alone is not a verified range |
| [2006–2023-named snapshot](snapshots/etf_data_2006_2023.csv) | Earlier experiment input retained for provenance |
| [Historical features](snapshots/etf_features.csv) | Original multiheader price/return features; archived scripts expect this format |

The canonical price loader consumes the simple price CSV, selects assets by name, checks positive finite prices and unique increasing dates, and builds features itself. It does not consume the multiheader feature export. Feature/target normalization is not imported from the historical pipeline.

The original downloader uses Yahoo Finance through `yfinance`. Historical library version, download timestamp, adjustment policy, and data-provider license were not recorded; exact historical downloads are therefore not guaranteed reproducible. New explicit downloads use adjusted closes and may differ from the snapshot. No API keys are required, but network availability and provider behavior are external dependencies.

**VIX caveat:** Yahoo’s ^VIX series represents an index level, not an investable security. An allocation to these index returns is an academic proxy only. No market-data ownership or blanket redistribution license is asserted by this repository; check Yahoo Finance/exchange/provider terms before downstream redistribution.