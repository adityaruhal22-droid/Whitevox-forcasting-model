# Whitevox Synthetic Forecasting Dataset

Synthetic (not real) monthly data modeled on Whitevox Digital — a Noida/Marlton-based AI-powered
digital marketing & reputation management agency offering SEO, GEO/AEO, ORM, social media, PPC,
digital PR, influencer marketing, celebrity management, and web/software development & staff
augmentation. It's calibrated loosely to public facts from whitevox.com (~9 years in business,
250+ clients, 90+ in-house experts, 3,000+ projects), but every number is generated, not sourced
from Whitevox's real financials.

## Files

### 1. `whitevox_monthly_by_service_line.csv` (1,040 rows)
One row per **service line × month**, Jan 2018 – Aug 2026 (104 months × 10 service lines).

| Column | Description |
|---|---|
| `date` | First of month |
| `service_line` | One of the 10 service lines (see below) |
| `active_clients` | Clients with an active engagement in that line, that month |
| `new_clients` | New clients signed that month |
| `churned_clients` | Clients lost that month |
| `new_leads` | Top-of-funnel leads generated for that service line |
| `revenue_usd` | Monthly revenue attributed to that service line |
| `marketing_spend_usd` | Whitevox's own promotional spend behind that service line |
| `projects_completed` | Projects/deliverables completed that month |
| `avg_deal_size_usd` | Average new-deal size that month |

Service lines and their (synthetic) launch dates — rows before launch are zeroed out, so each
line's real history starts only once it existed:
- SEO Services (2018-01), Online Reputation Management (2018-01), PPC / Paid Media (2018-01),
  Web & Software Development (2018-01)
- Social Media Marketing (2018-06)
- Digital PR (2019-01)
- Influencer Marketing (2019-06)
- Staff Augmentation (2020-01)
- Celebrity Management (2021-01)
- GEO / AEO (AI Search) Services (2024-03) — the newest offering, fast-growing off a small base

### 2. `whitevox_company_monthly_kpis.csv` (104 rows)
Company-wide roll-up, one row per month.

| Column | Description |
|---|---|
| `date` | First of month |
| `total_revenue_usd` | Sum of all service lines' revenue |
| `total_marketing_spend_usd` | Sum of all service lines' marketing spend |
| `total_new_clients` / `total_churned_clients` / `total_active_clients` | Client counts |
| `total_new_leads` | Sum of per-service leads |
| `total_projects_completed` | Sum of projects completed |
| `headcount` | In-house staff count (trends toward the site's "90+ experts") |
| `website_organic_sessions` | Whitevox's own organic search traffic |
| `avg_client_csat` | Average client satisfaction score (1–5) |
| `inbound_mqls` | Marketing-qualified leads into the top of funnel (pre-assignment) |
| `gross_margin_pct` | Blended gross margin |

## Patterns baked in (useful for testing a forecasting pipeline)

- **Trend**: each service line compounds at its own monthly growth rate, with mild saturation over time.
- **Seasonality**: Oct–Dec budget-flush bump, a smaller Jan bump, and a Jun–Jul dip — typical of B2B marketing-services demand.
- **Macro shocks**: a 2020 COVID dip/recovery, a 2022 ad-market slowdown, and a 2024+ "AI search" demand wave that lifts GEO/AEO sharply and gives the rest of the business a smaller lift.
- **New-service ramp**: GEO/AEO starts from zero in March 2024 and grows fast off a small base — good for testing cold-start / short-history forecasting.
- **Client funnel logic**: leads → new clients → active clients (net of churn) are internally consistent, so you can forecast at the funnel-metric level or the revenue level and cross-check.
- **Noise**: Gaussian noise layered on every series so no line is perfectly smooth.

## Suggested uses
- Monthly revenue forecasting (company-level, or per service line) with SARIMA/ETS/Prophet/XGBoost.
- Multi-series (hierarchical) forecasting: service-line forecasts should reconcile to the company total.
- Cold-start forecasting exercises using the GEO/AEO line (short history).
- Leading-indicator modeling: `new_leads`/`inbound_mqls` and `marketing_spend_usd` as predictors of future `revenue_usd`.
- Churn-adjusted client-count forecasting using `new_clients`/`churned_clients`.

## Caveats
This is entirely synthetic data generated with NumPy/pandas for prototyping a forecasting model.
It is not Whitevox's actual financial or operational data and shouldn't be presented as such.
