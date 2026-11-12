# Literature and data audit summary

## DECODE (Mishra, Lone & Mishra, 2024)

DECODE evaluates building energy prediction on I-BLEND using historical energy, occupancy, temperature, humidity and calendar features. Its preprocessing aligns one-minute energy, ten-minute occupancy and thirty-minute weather to a ten-minute table, uses time interpolation for missing weather, applies min-max normalization and splits chronologically 70:15:15. The comparison includes linear regression, decision tree, random forest and a one-layer 32-unit LSTM with two five-unit dense layers. RMSProp, batch size 64 and 20 epochs are reported for the LSTM. The paper reports average reference values around MAE 0.013 and R² 0.91 across seven buildings; these are cited reference results and are not copied into this system’s experiment output.

## I-BLEND (Rashid, Singh & Singh, 2019)

I-BLEND contains approximately 52 months of one-minute energy-related meter readings for seven campus buildings, ten-minute occupancy derived from aggregated Wi-Fi infrastructure records, an institute calendar, building metadata and supplementary weather. The data descriptor documents limited local logger weather coverage and nearby-station weather at half-hourly resolution. The platform therefore audits actual coverage and does not fill missing months or silently substitute a synthetic weather series.

## Research gap addressed by this repository

The operational limitation in using actual future occupancy is addressed with a separate occupancy forecast stage. The remaining extensions are evaluated as identifiable experiment configurations: energy-only vs predicted-occupancy vs oracle diagnostic, pooled leave-one-building-out transfer, few-shot adaptation, calibrated quantile intervals, controlled sensor corruption, and model-based (non-causal) explanations/scenarios.
