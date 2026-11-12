# Dataset preparation guide

## I-BLEND source

Obtain I-BLEND from the dataset citation associated with the Scientific Data article:

Rashid, Singh & Singh (2019), DOI `10.1038/sdata.2019.15`, data citation `10.6084/m9.figshare.c.3893581`.

The source files are not bundled in this repository. This preserves the dataset’s distribution terms and prevents accidental fabrication of research inputs.

## Recommended layout

```text
data/raw/i_blend/
  energy/
    Academic.csv
    Lecture.csv
    ...
  occupancy/
    Academic.csv
    Lecture.csv
    ...
  weather.csv
  calendar.csv
  metadata.csv                 # optional
```

The CLI also accepts a flat directory and reports candidate files; explicit API/ Python paths are preferred when filenames are ambiguous.

## Accepted canonical fields

After alias mapping, each table should expose:

- `timestamp`
- `building_id` for multi-building files, or an explicit building argument
- `energy_wh` or `power_w` for energy/power records
- `occupancy`
- `temperature_c`
- `humidity_pct`
- `working_day` and `high_activity` for calendar data

The loader recognizes common aliases such as `Timestamp`, `time`, `unix_timestamp`, `Power`, `power (watts)`, `Temperature`, `Humidity` and `Occupancy`. It does not guess physical units from magnitude. Record the source unit in the audit metadata and use an explicit conversion when needed.

## Explicit energy measurement configuration

Every energy audit/training request must declare the source semantics. For the published I-BLEND power files, a request may look like:

```json
{
  "source_unit": "W",
  "semantics": "instantaneous_power",
  "target_unit": "Wh",
  "source_column": "Power (watts)",
  "expected_interval_minutes": 1,
  "irregular_interval_policy": "use_actual_interval",
  "gap_factor": 1.5,
  "last_interval_policy": "drop"
}
```

This example is a configuration to be verified against the supplied file schema, not an automatic claim about every downloaded I-BLEND file. If a source is kW, Wh, kWh or cumulative energy, change `source_unit` and `semantics` explicitly. Power readings are integrated over actual timestamp durations; no conversion is based on numerical magnitude.

## Audit before training

```bash
campus-audit --root data/raw/i_blend --output artifacts/reports/i-blend-audit.json
```

Without `--source-unit` and `--semantics`, this command deliberately reports unresolved energy semantics. Add the explicitly verified source configuration before running research experiments.

Inspect timestamp coverage, duplicate counts, missing intervals, negative/implausible values, per-building coverage and the weather overlap. A weather-dependent experiment must be restricted to the valid overlap reported by the audit.

## Optional external validation

`campus_forecast.data.external.load_bdg2_energy` exposes an energy-only adapter for Building Data Genome 2. It does not fabricate or infer occupancy and its metrics must remain a separate external-dataset result. `load_uci_occupancy` supports a supplemental room-level occupancy experiment; it is not campus-scale occupancy and must not be merged into I-BLEND scores.
