# Failed pre-adaptive two-level relationship attempt

Status: archived, not a production result  
Failure time: 2026-09-05T20:46:44Z  
Campaign point: raw saturation `s0 = 0.25`  
Completed launch rays: 189 of 625

The fail-closed velocity-grid audit stopped at disc 7, point 6 because the
0.75 m/s trajectory had not reached a terminal classification by the 200 ms
deadline. Follow-up diagnostics showed that this trajectory escaped at
201.355 ms with matching 5, 2.5, and 1.25 microsecond integrations. The
production audit was therefore extended adaptively before the point was
restarted from ray zero. No final capture spectrum, loading rate, aggregate
row, or relationship figure was produced from this attempt.

The archive was moved intact out of the final campaign root. The completed
111-point force stage was not moved or modified.

## SHA-256 values captured before archival

- `statistics/campaign_metadata.json`: `B1A91A386B0405CB43B47ACBFAE673BAE58D0F90882F5EA07A59D9A0CA01F125`
- `statistics/launch_geometry.csv`: `02509217F582BC1619712CD31DE3FCB34AAC11B208C54A4CB228694F36603E17`
- `statistics/01_raw_saturation/points/000_s0_0p25/capture_endpoint_audit.csv`: `2F5F45C9C0136DDACD42C3C75BEC85FC232F31AFC384EBE1C8D6EDCC70350A29`
- `statistics/01_raw_saturation/points/000_s0_0p25/capture_velocity_overrides.json`: `0C8403A92CA65047284F7C5D43F8B2953DC4894A4C81631D53AD4B04D778DA6B`
- `statistics/01_raw_saturation/points/000_s0_0p25/capture_velocity_partial_samples.csv`: `B3E3CE866002342F1868D7F50115C71FF4B546353B029A1DA9DA4BD28525955A`
- `statistics/01_raw_saturation/points/000_s0_0p25/launch_geometry.csv`: `02509217F582BC1619712CD31DE3FCB34AAC11B208C54A4CB228694F36603E17`
- `statistics/01_raw_saturation/points/000_s0_0p25/run_metadata.json`: `F603339FF3A88DC762077A7967CA9A43C2243BE28B497C126B93FA37596CAE3C`

