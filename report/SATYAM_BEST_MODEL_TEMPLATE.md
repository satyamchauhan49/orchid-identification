# Best-model justification - fill only after real development comparisons

Choose using development data before opening test. Do not fill this with synthetic
numbers. Include a run/artifact ID for every claim.

> We selected [backbone/checkpoint] with [evidence variant] and [calibration settings]
> using [validation criterion, split hash and sample count]. Compared with our
> matched [baseline], it gave [top-1/macro-F1], [NLL/ECE] and [coverage/accepted accuracy]
> at frozen w=[...] and tau=[...]. The choice balances [measured discrimination],
> [calibration], [deferral], [measured compute cost if available] and [reviewed
> explanation quality; do not infer usefulness without a user study]. We preferred
> it over [alternative] because [measured development trade-off]. The independent
> locked test subsequently gave [metrics and grouped CIs], [paired difference CI]
> and [McNemar result with assumptions], which [did/did not] confirm this choice.
> Unsupported classes, [data caveats] and [missed targets] limit the conclusion.

The optional inference-cost field is left unmeasured until genuine outputs/models
are available. Do not rank models using test or claim any literature value is our result.
