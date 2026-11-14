# Free-Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Riaking89/free-model-aggregator-ria/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Riaking89/free-model-aggregator-ria/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

Ria's entry for the make.fun bounty [free-model-aggregator-01](https://make.fun/bounty/free-model-aggregator-01): one server (Node 22, standard library only) that serves the OpenAI Chat Completions API and the Anthropic Messages API — JSON and SSE — over ten simulated free-tier providers, answering every request with the best free model that can serve it.

## How it works

- Reads the trial brief (`MAKEFUN_PROMPT_FILE`): provider directory, sealed quality table, workload manifest.
- Probes each provider's catalog route in that provider's own wire (OpenAI, Anthropic, Gemini) and registers one catalog with the env (`POST $MAKEFUN_ENV_URL/register`).
- Routes each request to the highest-rated **free** model meeting its requirements (tools / vision / JSON mode / minimum context). An OpenAI request in JSON mode never rides an Anthropic wire, because that wire cannot carry `response_format`.
- Fails over on 402 / 429 / 5xx: a refusal that says the budget is spent (or the provider is out) marks that provider exhausted for the trial; a scheduled refusal does not, and the request is retried on the next candidate.
- Upstream calls are plain JSON even for streamed client requests; the server renders the client's stream itself, so the stream carries the provider's real usage (the client never asks for stream usage, so a passed-through stream would carry none).
- Polls `GET $MAKEFUN_ENV_URL/work` and exits 0 when the workload is done.

## Run

```sh
node main.ts --list        # print the catalog and exit 0
node main.ts              # serve a trial (needs MAKEFUN_ENV_URL + MAKEFUN_PROMPT_FILE)
```

No dependencies, no API keys, no network beyond `$MAKEFUN_ENV_URL`.
