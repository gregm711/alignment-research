# When does a circuit need to communicate?

Three local experiments separate the value of a message, the cost of figuring out who originated it, and the use of an intermediate result by another circuit.

| Question | Controlled comparison | Result |
|---|---|---|
| Can twenty echoes drown independent evidence? | Same packets, checks and answers at 48 cells; discard the supplied source identity only when aggregating. | Accuracy fell from 82.0% to 59.7% on the copied-source panel. When the copied reading was wrong it fell from 75.1% to 6.6%. The learned and analytical selective policies chose identical actions. [Pooling](pooling/README.md) |
| Can the receiver infer common origin without being told? | Infer redundancy from eight unlabeled events; compare on identical checks, and switch the upstream group without warning. | On fixed groups the receiver gained 20.37 points over voting at equal receipts. On fresh changing groups, updating on disagreement cost 1.79 µJ per decision versus 10.24 µJ for updating every event, with 1.25 points lower accuracy. Updating every eighth event performed similarly to the gate. [Hidden sources](hidden-sources/README.md) |
| Does recruitment execute a composition? | Keep all worker addresses and contacts fixed; replace the argument of the last function. Train at depth one and two, then test fresh depth-five tasks. | At 640 available workers accuracy fell from 98.82% to 5.97%. The conventional router reached 98.78% at lower CPU cost. The functions and available connections were supplied. [Composition](composition/README.md) |

On the communication test, extending from 48 to 96 cells also exposed the cost of stopping too early. The learned selective policy saved 74.2% of bits against full-format broadcast at 96 cells and lost 4.82 points of accuracy. Neither adding workers nor multiplying local checks guarantees a better global decision. On the function test, width increased available modules while the required sequential depth stayed fixed. All studies use synthetic worlds; independent source discovery, operation invention and hardware scaling on an actual network remain open beyond these settings.

Each study links its declared predictions, paired comparisons, cost assumptions, raw outputs and reproduction command. Timings measure batched inference on one Apple Silicon laptop. The meter reads the process's kernel-attributed CPU energy in nanojoules. It excludes remote transmissions, physical sensor acquisition, GPU use and the rest of the device. Backend work, vectorization and other host activity can affect timings. Simulated network bits and external checks are reported separately. [Measurement implementation](meter.py) · [Apple XNU counter documentation](https://github.com/apple-oss-distributions/xnu/blob/main/doc/observability/recount.md)

## Transfer beyond planted errors

The earlier 6.6% to 75.1% conditional result used supplied source IDs. On a fresh hidden-ID synthetic panel, the receiver inferred redundancy from eight unlabeled readings per relay. When the near-copy ancestor was wrong, bit accuracy rose from zero to 61.9% without checks, or 75.0% with matched optional independent evidence. When the ancestor was right, downweighting cost 8.0 points with matched checks. [Hidden-source audit](hidden-sources/ancestor-results.json)

| New test | What the comparison settled |
|---|---|
| Clustered changes over long quiet periods | At equal rebuild count, the disagreement gate beat the fixed-clock frontier by 0.77 points on replication. At arrival-by-arrival deadlines, an eight-event clock reached 77.33% at 1.92 µJ per decision vs the gate's 77.12% at 2.58 µJ. [Bursty replication](bursty/README.md) |
| Two actual instrument-array recordings | The gate scored 77.3% and 69.6% balanced accuracy. One sensor selected on earlier data scored 80.6% and 85.2%. Labels identify changes in commanded gas set points; common origin is unobserved. [Physical sensors](real-sensors/README.md) |
| Two CDC FluSight seasons | In 1,683 held-out state-weeks the gate correctly predicted the direction of admissions 72.85% of the time, majority 75.40%, and the mean numeric forecast 76.95%. [Archived forecasts](real-forecasts/README.md) |
| 18 archived LLMBar judge configurations | Continuously estimated redundancy corrected 161/319 vs 139/319 for majority, while the single judge selected on a separate calibration set solved 252/319. Twenty injected copies per question failed to produce the registered five-point lift over majority. The corpus has no deployment chronology. [Model judgements](real-judges/README.md) |

An unlabeled agreement pattern admits several explanations, including a common error and several observers correctly following the same change. Outcome feedback can distinguish useful witnesses when agreement alone cannot. These results support retaining the source of an observation when available, measuring competence where independent outcomes exist, and testing the additional value of another observation for a particular decision. A dedicated learned routing or provenance module has yet to earn a general accuracy or energy advantage across these comparisons.
