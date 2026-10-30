# Data governance

## 1. Data inventory
| Dataset | Source | Owner | Contains personal data? | Retention |
|---|---|---|---|---|
| Audio clips (5 s) | House microphones | Farm (data controller) | Possibly, as incidental staff speech | Raw audio 30 days; derived features and predictions for the project period + 1 year |
| Recording metadata | Devices / ingestion | Farm | No | Project period + 1 year |
| Model predictions and alerts | FlockWatch | Research team (processor) | No | Project period + 1 year |
| Flock record cards | Farm management | Farm | No (staff initials removed on ingest) | As the farm's own policy |
| Public datasets | Mendeley Data, CC BY 4.0 | Original authors | No | Kept with attribution |

## 2. Roles and access
- **Farm manager:** sees their own houses only (enforced by the `house_id`/`farm` filter in the dashboard and API).
- **Vet:** reads alerts and record cards for the farms they serve.
- **Researchers:** pseudonymised farm and house IDs. No raw audio leaves the farm environment without agreement.
- **Data steward:** manages quality rules, retention jobs and the audit log.

## 3. Quality rules at collection (`src/flockwatch/ingest.py`)
| Check | Rule | Action |
|---|---|---|
| Required metadata | clip_id, house_id, device_id, recorded_at | Quarantine |
| Sample rate | at least 16 kHz | Quarantine |
| Duration | 4.5–5.5 s | Quarantine |
| Clipping | no more than 1% of samples at full scale | Quarantine (microphone overload) |
| Silence | RMS at least 0.001 | Quarantine (possible device fault) |
| Duplicates | SHA-256 of samples | Quarantine |

Quarantined clips are **kept with a reason**, never silently dropped, so data stewards can trace device faults. Sensor coverage (clips received against clips expected) is monitored per house per day to detect outages.

## 4. Lineage and audit
- Each recording stores a SHA-256 fingerprint, its source and its ingestion time.
- Each prediction stores the model version and the prediction time.
- Each pipeline step (simulation, ingestion, scoring, alerting) is written to `audit_log` with a timestamp.
- Models are versioned and their metrics stored in `models/metrics.json`.

## 5. Privacy (UK GDPR)
- **Minimisation:** only 5-second clips are kept, and raw audio is deleted after 30 days. Only features and predictions are retained long term.
- **Speech:** the next step is a voice-activity filter that discards clips containing human speech before storage.
- **Transparency:** staff are told about the microphones and their purpose (signage and a privacy notice).
- **Security:** the database is stored encrypted at rest in deployment. Access is through authenticated roles.

## 6. Reporting to stakeholders
| Audience | Format | Frequency | Content |
|---|---|---|---|
| Farm manager | Dashboard + weekly Excel report | Daily / weekly | House status (red/amber/green), mortality, weight, record cards |
| Vet | Dashboard early-warning view, alert notification | On alert | Sound index against baseline, water intake, alert history |
| Researchers | API + database export | On demand | Clip-level predictions, metrics, pseudonymised |
| Funder / project board | Summary report | Quarterly | Detection lead time, false-alert rate, data coverage |
