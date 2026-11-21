# Updated ConvNeXt validation resolver handoff

Source: convnext_tiny_lr0.0001_handoff.zip supplied by Satyam. All 18 inventory hashes passed. Exact validation IDs, reported overall/accepted correctness counts and exchange artifact/bundle/map hashes were checked. No test arrays were opened.

| Metric | Earlier local policy | Updated Satyam policy |
|---|---:|---:|
| Full-coverage accuracy | 79.58% | 79.42% |
| Accepted accuracy | 96.86% | 94.46% |
| Deferral | 39.98% | 30.48% |

The updated policy fits fusion weight and threshold using disjoint calibration groups. It accepts 885/1,273 images, of which 836 are correct; defers 388; produces 877 trusted_agreement and 8 auto_resolved cases. Overall 1,011/1,273 proposed predictions are correct. Accepted error is 49/885 = 5.54%, above the 1.5% target.

Weight 1.0 means full-coverage fusion equals the DNN, so auto_resolved stage names do not demonstrate any BN correction or accuracy gain. Neither validation risk nor deferral targets are established. These are model-selected validation results, not final test findings.

Earlier site/real_validation remains a separate policy experiment. The new exchange includes 1398.jpg, which is absent from the downloaded earlier examples. A complete updated visual release needs the original Colab images and matching checkpoint; do not fabricate or reuse unrelated heatmaps.

Next run notebooks/Sneha_Satyam_Handoff.ipynb in the existing Colab runtime to verify the handoff and generate ten updated four-layer records, including the actual auto_resolved case. Frozen test evaluation remains a separate team stage.

## Updated visual delivery
Imported and checked site/satyam_validation with ten real validation records: three trusted, six deferred and one auto-resolved. Record provenance matches Satyam's supplied frozen artifact. Original inputs, all three head overlays and candidate references are present. Website links and payload were checked. The earlier site/real_validation remains unchanged as a separate policy experiment.
