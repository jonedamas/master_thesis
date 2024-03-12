# Database Status

## news

Unaltered news articles, gathered by news article APIs.

Number of rows: **27893**.

Unique sources: **TG, NYT**.

Unique queries: **CrudeANDOil, NaturalANDGas**.

| datetime                   | article_id                                                                                                       | headline                                                                                             | web_url                                                                                                                                      | source   | query       |
|:---------------------------|:-----------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:---------|:------------|
| 2023-12-31 07:00:56.000000 | business/2023/dec/31/smiles-all-round-as-financial-markets-end-2023-on-an-unexpected-high                        | Smiles all round as financial markets end 2023 on an unexpected high                                 | https://www.theguardian.com/business/2023/dec/31/smiles-all-round-as-financial-markets-end-2023-on-an-unexpected-high                        | TG       | CrudeANDOil |
| 2023-12-28 16:02:15.000000 | business/live/2023/dec/28/pound-dollar-uk-economy-predicted-to-turn-corner-2024-ftse-stock-markets-business-live | UK dealmaking shrinks in 2023, but economy predicted to ‘turn corner’ in 2024 – as it happened       | https://www.theguardian.com/business/live/2023/dec/28/pound-dollar-uk-economy-predicted-to-turn-corner-2024-ftse-stock-markets-business-live | TG       | CrudeANDOil |
| 2023-12-27 16:17:09.000000 | business/live/2023/dec/27/stock-markets-santa-rally-soft-landing-hopes-ftse-100-wall-street-business-live        | New York Times accuses ChatGPT maker OpenAI and Microsoft of copyright infringement – as it happened | https://www.theguardian.com/business/live/2023/dec/27/stock-markets-santa-rally-soft-landing-hopes-ftse-100-wall-street-business-live        | TG       | CrudeANDOil |

## news_filtered

Filtered news articles through NaN removal, clustering and duplicate removal.

Number of rows: **27281**.

Unique sources: **TG, NYT**.

Unique queries: **CrudeANDOil, NaturalANDGas**.

| datetime                   |   Unnamed: 0 | article_id                                                                                                       | headline                                                                                             | web_url                                                                                                                                      | source   | query       |
|:---------------------------|-------------:|:-----------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:---------|:------------|
| 2023-12-31 07:00:56.000000 |            0 | business/2023/dec/31/smiles-all-round-as-financial-markets-end-2023-on-an-unexpected-high                        | Smiles all round as financial markets end 2023 on an unexpected high                                 | https://www.theguardian.com/business/2023/dec/31/smiles-all-round-as-financial-markets-end-2023-on-an-unexpected-high                        | TG       | CrudeANDOil |
| 2023-12-28 16:02:15.000000 |            1 | business/live/2023/dec/28/pound-dollar-uk-economy-predicted-to-turn-corner-2024-ftse-stock-markets-business-live | UK dealmaking shrinks in 2023, but economy predicted to ‘turn corner’ in 2024 – as it happened       | https://www.theguardian.com/business/live/2023/dec/28/pound-dollar-uk-economy-predicted-to-turn-corner-2024-ftse-stock-markets-business-live | TG       | CrudeANDOil |
| 2023-12-27 16:17:09.000000 |            2 | business/live/2023/dec/27/stock-markets-santa-rally-soft-landing-hopes-ftse-100-wall-street-business-live        | New York Times accuses ChatGPT maker OpenAI and Microsoft of copyright infringement – as it happened | https://www.theguardian.com/business/live/2023/dec/27/stock-markets-santa-rally-soft-landing-hopes-ftse-100-wall-street-business-live        | TG       | CrudeANDOil |
