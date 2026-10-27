# Jev Reranker Adapter

> Route ai-memory's LLM reranker to a Jev judge endpoint while keeping the
> hosted LLM for everything else. Stdlib-only Python, no Rust changes.

[`AI_MEMORY_RERANKER=llm`](llm-providers.md) reorders `memory_query`
candidates with the configured chat provider. That works until the provider
is a hosted reasoning model: the reranker prompt is long, a graded judgement
over up to 30 candidates, and a slow model answers in tens of seconds.
That is longer than one `memory_query` should ever take, and long enough to trip the
server's completion timeout, which turns the reranker into a dead feature
that stalls every query before falling back to the original order.

The reranker prompt, though, is already a scoring rubric: grade each
candidate 1.0 (direct answer) / 0.7 (same topic) / 0.3 (tangential) /
0.0 (unrelated). A judge endpoint that scores a fixed rubric answers the
same question without autoregressive decoding. The rubric adapter
described here sits between ai-memory and the provider, translates exactly
that request into one batched Jev `score` call, and proxies everything else
unchanged. Its script, `jev_rerank_shim.py`, is no longer in the tree (it
was removed in commit `e45db18a`; recover it with
`git show 311fe87f:docs/examples/jev-reranker-adapter/jev_rerank_shim.py`).
The adapter that ships under
[`docs/examples/jev-reranker-adapter/`](examples/jev-reranker-adapter/) is
the [choice-contrastive variant](#choice-contrastive-variant). The same
directory also holds a choice adapter for the OpenAI Decisions API, described
in [OpenAI Decisions variant](#openai-decisions-variant).

## How it works

ai-memory 2.4.0 has a single chat-provider configuration, so the reranker
and consolidation share one base URL. The adapter does not try to split
configuration; it splits traffic by request shape:

1. The reranker's system prompt starts with the fixed sentence
   `You are a retrieval reranker for a software project's memory wiki.`
   (see `crates/ai-memory-llm/src/reranker.rs`). Requests whose system
   message has that prefix are reranker requests.
2. The user message is `{"query", "candidates": [{"candidate", "title",
   "text"}]}` with 1-based indices. The adapter renders one `state` block
   (query + numbered candidates) and one `score` question per candidate
   whose `criteria` mirror the reranker prompt's four grades, then makes a
   single POST to the Jev `/v1/systemone` endpoint.
3. Each answer's rubric index maps back to `relevance` 0.0 / 0.3 / 0.7 /
   1.0 (index / 3), and the adapter returns
   `{"scores": [{"candidate": n, "relevance": f}]}` as plain chat-completion
   content, which is the exact shape the reranker's tolerant parser expects.
4. Every other request (consolidation, lint, bootstrap, plain chats) is
   reverse-proxied to the real upstream byte-for-byte, `Authorization`
   forwarded verbatim. The adapter stores no secrets.

Failure semantics match the server contract: if the Jev call fails, the
adapter answers HTTP 500 and ai-memory keeps its own candidate order, the
same as any provider outage. A judge endpoint that is down degrades to
"no reranking", never to "no search".

The choice variant also fails closed on a *malformed* answer: an HTTP 200
whose `probabilities` object is missing, renamed, short of candidates,
padded with extras, or holds booleans, NaN/Inf, out-of-range numbers, or a
distribution nowhere near normalised is rejected with 500 as well. The
older shim defaulted those to `0.0`, which produced a well-formed but
meaningless score set the server could not distinguish from a real
judgement. `test_jev_rerank_shim_choice.py` (stdlib `unittest`, no backend
needed) pins each rejection path.

One cosmetic note: the server logs still show the provider's configured
model name for reranking (it comes from provider config, not from the
adapter's reply), so the reranker leg will be attributed to your hosted
model in `docker logs` even while the adapter serves it.

## Deploy

The adapter is stdlib-only Python 3. Configure with environment variables
(all defaults are loopback examples):

| Env | Meaning | Default |
|---|---|---|
| `JEV_URL` | Jev `/v1/systemone` endpoint | `http://127.0.0.1:18095/v1/systemone` |
| `JEV_MODEL` | model name sent to Jev | `jev-latest` |
| `UPSTREAM` | real OpenAI-compat base URL | `http://127.0.0.1:8000` |
| `LISTEN` | adapter bind address | `127.0.0.1:18097` |

Run it next to the server (systemd unit adapted to your paths):

```ini
[Unit]
Description=ai-memory Jev reranker adapter
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /opt/ai-memory/ops/jev_rerank_shim_choice.py
Environment=JEV_URL=http://127.0.0.1:18095/v1/systemone
Environment=UPSTREAM=http://127.0.0.1:8000
Environment=LISTEN=127.0.0.1:18097
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Then point the provider at the adapter instead of the upstream:

```console
AI_MEMORY_LLM_PROVIDER=openai-compat
AI_MEMORY_LLM_BASE_URL=http://127.0.0.1:18097/v1
AI_MEMORY_RERANKER=llm
```

The unit runs the shipped choice-contrastive variant. To run the rubric
adapter instead, recover `jev_rerank_shim.py` from git history and point
`ExecStart` at it. Everything else (env, unit, provider config) is
identical.

Verify the split with `journalctl -u <unit>`: reranker requests log
`jev-choice N candidates in X.XXXs` (the rubric adapter logs
`jev N candidates in X.XXXs`), everything else is silent (proxied).

## Benchmarks

Offline A/B, same candidate pool per query, 102 queries / 99 scored against
a hand-curated golden set for a real production wiki (FTS5 + vector + graph
hybrid, `explain=true` score details, identical server build):

| Reranker | hit@1 | MRR | NDCG@10 | mean rerank latency |
|---|---|---|---|---|
| none (server order) | 0.495 | 0.651 | 0.739 | — |
| hosted model A (reasoning, max) | 0.778 | 0.840 | 0.875 | **20.2 s** |
| hosted model B (reasoning) | 0.768 | 0.834 | 0.870 | 13.9 s |
| **Jev via this adapter** | **0.778** | **0.838** | **0.873** | **0.205 s** |

The judge endpoint matches the hosted reasoning models' quality at
~100× lower latency. Hosted model A's 20.2 s mean sat exactly on the
server's 20 s completion timeout, so in production every rerank call timed
out and the feature was effectively off (queries stalled 20 s, then fell
back to the original order).

Live end-to-end (production server, `AI_MEMORY_RERANKER=llm` through the
adapter, full golden set): hit@1 0.657 / MRR 0.788 / NDCG@10 0.843,
0 errors in 102 queries, mean `memory_query` latency 2.2 s (p50 2.21 s,
max 3.07 s), 110/110 rerank translations served, 0 adapter failures. The
gap to the offline arm is candidate-pool shape, not scoring: the live
server over-fetches 15–30 candidates with its own bounded snippets (the
offline arm rescored a fixed top-10 pool), and in 32 of 34 non-top-1
queries the expected page was still returned, mostly at rank 2, with
hit@5 at 0.949. Versus the no-reranker baseline that is +16.2 points
hit@1 and +10.4 points NDCG@10 end to end.

Judge latency scales with the candidate count in the batch (the rubric
question is asked once per candidate): ~0.2 s for 10 candidates, ~1.6 s
mean for the 15–30-candidate batches the live server sends, still an order
of magnitude under any hosted reasoning model.

## Choice-contrastive variant

[`jev_rerank_shim_choice.py`](examples/jev-reranker-adapter/jev_rerank_shim_choice.py)
flips the question shape: instead of one rubric `score` per candidate, it
asks a single `choice` question over the whole candidate list and maps each
candidate's choice probability directly to `relevance`. The production
wording asks which specific page contains the answer, and tells the model
not to pick an index, catalog, or summary when a more specific page answers.
On a 30-candidate pool that wording moved the 35B from 0.828 to 0.879 hit@1
(5 fixed, 0 broken) versus "which document is most relevant?". The table
below is the earlier choice-versus-rubric comparison, not that wording test.

That is deliberately the substitution the caveat below used to warn about.
It is sound here because the reranker's consumer is sort-only: the
server reorders candidates by `relevance` and never thresholds or sums its
absolute value, and a per-question monotonic map preserves order exactly.
The old warning applies to consumers that *read* absolute relevance
grades; the reranker leg does not, and the caveat below is now scoped to
that case.

Measured on one production deployment (same 102-query golden set, same
live server and backend, end-to-end through the adapter):

| Backend model | Adapter | hit@1 | NDCG@10 | mean rerank latency |
|---|---|---|---|---|
| 35B (Qwen3.6-35B-A3B) | rubric `score` | 0.636 | 0.833 | 2.2 s |
| 35B (Qwen3.6-35B-A3B) | **choice** | **0.778** | **0.893** | **0.75 s** |
| 4B replay-trained | rubric `score` | 0.596 | 0.776 | 8.2 s |
| 4B replay-trained | **choice** | **0.808** | **0.895** | 1.16 s |

Two takeaways. First, the choice shape beat the rubric shape by 14–21
hit@1 points *on the same backend model*: relative comparison is an easier
judgement than absolute grading, and it shows up in both quality and
latency (one batched question for the whole list instead of one question
per candidate). Second, the rubric shape is where replay-trained small
models collapse (0.596 on 4B vs 0.636 on 35B); in the choice shape the
same 4B overtakes the 35B. If your backend is a small replay-trained
judge, the choice adapter is the difference between usable and not.

Prefer the choice adapter when the reranker leg is the only consumer.
Keep the rubric adapter if anything downstream reads absolute relevance
grades (thresholding, logging heuristics, score fusion), because choice
probabilities carry no absolute meaning and their scale shifts with
candidate count.

## OpenAI Decisions variant

[`decisions_rerank_shim.py`](examples/jev-reranker-adapter/decisions_rerank_shim.py)
asks the same single `choice` question of the
[OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)
(`POST /v1/decisions`, model `gpt-6-luna`) instead of Jev. It differs from
the Jev choice adapter in four ways:

- The query and candidates travel as one `input_json` part, which the API
  treats as data, and each choice's description is the page title.
- It holds no secret. It reuses the `Authorization` header ai-memory already
  sends, so the OpenAI key that serves consolidation also serves reranking.
- A `refusal` answer is a failure like a malformed one: the adapter answers
  500 and ai-memory keeps its own order.
- The question is neutral: "Which candidate page is most relevant to the
  query?". With the Jev adapter's wording, Decisions refused 10 and 11 of
  470 queries in two runs, at the same hit@k; the neutral wording refused 2.

The provider stays `openai`, pointed at the adapter. The `openai` provider
sends OpenAI's own request dialect whatever the base URL, so consolidation
requests reach the upstream unchanged:

```toml
llm_provider = "openai"
llm_model = "gpt-6-luna"
llm_base_url = "http://127.0.0.1:18098"
reranker = "llm"
```

| Env | Meaning | Default |
|---|---|---|
| `DECISIONS_URL` | Decisions endpoint | `https://api.openai.com/v1/decisions` |
| `DECISIONS_MODEL` | model sent to Decisions | `gpt-6-luna` |
| `DECISIONS_TIMEOUT` | seconds per judge call | `15` |
| `UPSTREAM` | where every other request goes | `https://api.openai.com` |
| `LISTEN` | adapter bind address | `127.0.0.1:18098` |

Every LLM request now passes through the adapter, so consolidation stops
when the adapter does. Under systemd, give the server unit
`Wants=` and `After=` on the adapter's unit.

### Measured on LongMemEval-S

All 470 questions, zero-LLM retrieval (FTS5 with the default observation
stream), server and harness built from `main` at `0e548807` on an AMD Ryzen 7
8700G. `--concurrency 4` keeps at most four queries in flight, so the
reranker's four-call gate never skips a query. Each reranked query sent
about 3,000 input tokens to Decisions (mean 3,035, max 4,962), roughly
$0.0003 at the published uncached rate.

| reranker | hit@1 | hit@3 | hit@5 | hit@10 | p50 | p95 | failed calls |
|---|---|---|---|---|---|---|---|
| none | 0.857 | 0.930 | 0.951 | 0.972 | 11 ms | 81 ms | none |
| Decisions, neutral wording (shipped) | 0.906 | 0.968 | 0.983 | 0.989 | 387 ms | 767 ms | 2 refusals |
| Decisions, Jev wording, run 1 | 0.900 | 0.970 | 0.979 | 0.985 | 381 ms | 641 ms | 11 refusals |
| Decisions, Jev wording, run 2 | 0.909 | 0.972 | 0.981 | 0.987 | 371 ms | 638 ms | 10 refusals |
| `openai` chat, `gpt-6-luna` | 0.902 | 0.981 | 0.987 | 0.991 | 6,331 ms | 13,262 ms | 6 past the 20 s budget, one HTTP 500 |

The two runs of the same Decisions wording differ by 0.009 at hit@1, so
read differences below 0.01 as noise. Decisions matches the chat
reranker at hit@1 at about a sixteenth of its median latency. The chat
reranker leads at hit@3 and hit@5. Decisions reports probabilities to two
decimals, so every candidate below 0.005 ties at 0.00 and keeps its fused
order: only the head of the list is reordered. Both judges cost hit@1 on
the knowledge-update slice (0.986 without reranking, 0.931 with the chat
reranker, 0.917 to 0.931 across the three Decisions runs), where an older
session that states the superseded value sometimes wins.

To reproduce, run the adapter on `127.0.0.1:18098`, then:

```bash
ai-memory-eval retrieval --server-bin target/release/ai-memory \
  --concurrency 4 --http-timeout-secs 900 \
  --server-env AI_MEMORY_RETRIEVAL__OBSERVATION_STREAM=true \
  --server-env AI_MEMORY_EVAL_SKIP_WIKI_CHECKPOINTS=1 \
  --candidate-reranker \
  --candidate-server-env AI_MEMORY_RETRIEVAL__OBSERVATION_STREAM=true \
  --candidate-server-env AI_MEMORY_EVAL_SKIP_WIKI_CHECKPOINTS=1 \
  --candidate-server-env AI_MEMORY_LLM_PROVIDER=openai \
  --candidate-server-env AI_MEMORY_LLM_MODEL=gpt-6-luna \
  --candidate-server-env AI_MEMORY_LLM_BASE_URL=http://127.0.0.1:18098 \
  --candidate-server-env OPENAI_API_KEY="$OPENAI_API_KEY"
```

Until the harness masks credential values (EVALS_ISSUE_REF), the report's
config line prints that key: remove it before sharing the report.

## Caveats

- The adapter keys on the exact system-prompt prefix above. If the
  reranker prompt wording changes in a future release, detection (not
  scoring) breaks first: reranker requests would be proxied to the hosted
  model, which is the pre-adapter behavior, not an outage.
- Rubric scoring produces absolute relevance grades. If anything
  downstream of `relevance` reads absolute values (thresholding, fusion,
  logged heuristics), keep the rubric adapter: per-candidate choice
  probabilities from the choice adapter are normalized across the whole
  candidate list (winner ≈ 0.99, rest ≈ 0.001, scale shifts with
  candidate count) and carry ordering information only. The sort-only
  reranker leg itself reads no absolute values, which is why the choice
  variant is safe to substitute for it.
- Run the adapter on loopback or a trusted private network. It forwards
  `Authorization` headers verbatim and adds no authentication of its own.
- The Decisions adapter sends each reranked query and up to 30 candidate
  titles and snippets to OpenAI, as the `llm` reranker already does with
  any hosted provider. See the reranking entry in `SECURITY.md`.
