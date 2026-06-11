// Drives the provider through the real AI SDK (`embed`/`embedMany`) against the live Forge API.
// Live tests skip without FORGE_API_KEY (honest skip, no faking).
import { test } from "node:test";
import assert from "node:assert/strict";
import { embed, embedMany } from "ai";
import { createForge } from "../src/index.js";

const KEY = process.env.FORGE_API_KEY;
const BASE = process.env.FORGE_BASE_URL || "https://api.voxell.ai";
const skip = KEY ? false : "set FORGE_API_KEY to run live tests";
const forge = createForge({ apiKey: KEY, baseURL: BASE });

test("model shape (no network)", () => {
  const m = createForge({ apiKey: "x" }).textEmbeddingModel("turbo");
  assert.equal(m.specificationVersion, "v3");
  assert.equal(m.modelId, "turbo");
  assert.equal(m.provider, "forge.embedding");
  assert.equal(m.supportsParallelCalls, true);
});

test("embedMany maps Forge request and response fields without network", async () => {
  const calls: Array<{ input: string; init: RequestInit }> = [];
  const fetchImpl = (async (input, init) => {
    calls.push({ input: String(input), init: init ?? {} });
    return new Response(JSON.stringify({ embeddings: [[1, 2], [3, 4]], tokens: 9 }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }) satisfies typeof fetch;

  const provider = createForge({
    apiKey: "secret",
    baseURL: "https://forge.test///",
    headers: { "x-static": "yes" },
    fetch: fetchImpl,
  });

  const result = await embedMany({
    model: provider.textEmbeddingModel("pro"),
    values: ["alpha", "beta"],
    providerOptions: { forge: { inputType: "query", dimensions: 2 } },
    headers: { "x-call": "1" },
  });

  assert.deepEqual(result.embeddings, [
    [1, 2],
    [3, 4],
  ]);
  assert.deepEqual(result.usage, { tokens: 9 });
  assert.deepEqual(result.warnings, []);
  assert.equal(calls.length, 1);
  const [call] = calls;
  const requestHeaders = call.init.headers as Record<string, string>;
  assert.equal(call.input, "https://forge.test/v1/embed");
  assert.equal(call.init.method, "POST");
  assert.deepEqual(
    {
      "content-type": requestHeaders["content-type"],
      authorization: requestHeaders.authorization,
      "x-static": requestHeaders["x-static"],
      "x-call": requestHeaders["x-call"],
    },
    {
      "content-type": "application/json",
      authorization: "Bearer secret",
      "x-static": "yes",
      "x-call": "1",
    },
  );
  assert.match(String(requestHeaders["user-agent"]), /^ai\//);
  assert.deepEqual(JSON.parse(String(call.init.body)), {
    texts: ["alpha", "beta"],
    model: "pro",
    input_type: "query",
    dim: 2,
  });
});

test("embed surfaces Forge JSON error messages without network", async () => {
  const provider = createForge({
    apiKey: "secret",
    fetch: async () =>
      new Response(JSON.stringify({ error: "quota exceeded" }), {
        status: 429,
        statusText: "Too Many Requests",
        headers: { "content-type": "application/json" },
      }),
  });

  await assert.rejects(
    embed({ model: provider("turbo"), value: "hello" }),
    /Forge embeddings: HTTP 429 quota exceeded/,
  );
});

const DIMS: Array<[string, number]> = [
  ["turbo", 1024],
  ["pro", 2560],
  ["ultra", 4096],
];
for (const [model, dim] of DIMS) {
  test(`embed via AI SDK: ${model} → ${dim}-d`, { skip }, async () => {
    const { embedding } = await embed({ model: forge.textEmbeddingModel(model), value: "why forge" });
    assert.equal(embedding.length, dim);
    assert.ok(embedding.every((n) => Number.isFinite(n)));
  });
}

test("embedMany: 2 values → 2 vectors + usage", { skip }, async () => {
  const { embeddings, usage } = await embedMany({
    model: forge.textEmbeddingModel("turbo"),
    values: ["alpha", "beta"],
  });
  assert.equal(embeddings.length, 2);
  assert.equal(embeddings[0].length, 1024);
  assert.ok((usage?.tokens ?? 0) > 0);
});

test("Matryoshka via providerOptions.forge.dimensions → 256", { skip }, async () => {
  const { embedding } = await embed({
    model: forge.textEmbeddingModel("turbo"),
    value: "truncate me",
    providerOptions: { forge: { dimensions: 256 } },
  });
  assert.equal(embedding.length, 256);
});

test("default model (forge()) is turbo/1024", { skip }, async () => {
  const { embedding } = await embed({ model: forge(), value: "default" });
  assert.equal(embedding.length, 1024);
});
