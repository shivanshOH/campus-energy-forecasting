# Methodology, assumptions and research boundaries

## DECODE foundation

The primary methodological reference is Mishra, Lone and Mishra, *DECODE*, Energy and Buildings (2024), DOI 10.1016/j.enbuild.2024.113950. The paper uses historical energy, occupancy, temperature, humidity and calendar features, a chronological 70:15:15 split, min-max normalization and comparisons among linear regression, decision tree, random forest and a 32-unit LSTM. The LSTM is reported with RMSProp, batch size 64 and 20 epochs, followed by two dense layers of five units each.

The paper does not fully specify every production detail needed for exact reproduction, including all window boundaries, exact energy aggregation semantics, missing-row policy, random seeds and checkpoint selection. This implementation therefore calls its model **DECODE-style**, records each assumption in experiment metadata, and does not claim exact reproduction.

## I-BLEND facts relevant to this implementation

The I-BLEND data descriptor reports seven buildings: Academic, Lecture, Library, Dining, Facilities, Boys dormitory and Girls dormitory. Energy files contain Unix timestamps and power-related measurements at a released one-minute alignment. Occupancy is supplied at ten-minute resolution. Weather coverage includes nearby-station half-hourly observations and a four-month local logger, rather than guaranteed full 52-month building-aligned weather coverage. Calendar data is date-based.

## Common grid and aggregation assumptions

- Default grid: ten minutes, matching occupancy resolution.
- Energy: mean of valid source measurements within each grid bin by default. This is a power-like aggregation and can be changed in configuration; it is not asserted to be the paper’s hidden implementation.
- Occupancy: maximum in each ten-minute bin, matching the I-BLEND descriptor’s documented downsampling procedure.
- Weather: time-aware alignment/resampling only within observed coverage; no synthetic values are inserted.
- Calendar: date join and forward use of only known calendar fields.

The current implementation requires an explicit `EnergyMeasurementConfig` before loading source energy files. The configuration declares whether the source is W/kW instantaneous power or Wh/kWh interval/cumulative energy, the target unit, expected interval, irregular-gap policy and final-interval handling. W/kW readings are integrated using the actual timestamp-to-next-timestamp duration; interval energy is summed; cumulative meters are differenced with reset rows rejected. Numeric magnitude is never used to infer units.

## Operational information boundary

At issue time `t`, a deployment forecast can use observations at or before `t`, known future calendar attributes, and an archived weather forecast if one is explicitly supplied as such. It cannot use observed future weather or future occupancy. The oracle occupancy configuration exists only as an upper-bound diagnostic and is rejected by operational prediction APIs unless explicitly marked diagnostic.

## Report language

Measured values, model predictions, calibrated empirical intervals and model-based what-if scenarios are stored as separate concepts. Prediction intervals are empirical/calibrated on the selected calibration data and are not guarantees. Scenario changes are not causal estimates or evidence of actual energy savings.
