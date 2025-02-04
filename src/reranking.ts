import type { Context } from "hono";
import { loadModel } from "./models";
import type { RankedResult } from "./types";

export async function handleReranking(c: Context) {
  try {
    const { model, query, documents } = await c.req.json();

    if (!model || !query || !Array.isArray(documents)) {
      return c.json(
        {
          error: {
            message:
              "Missing required parameters: model, query, documents (array)",
            type: "invalid_request_error",
            param: !model ? "model" : !query ? "query" : "documents",
            code: null,
          },
        },
        400
      );
    }

    const modelContext = await loadModel(model, "reranker");
    if (!modelContext.rankingContext) {
      return c.json(
        {
          error: {
            message: "Model is not suitable for reranking",
            type: "invalid_request_error",
            param: "model",
            code: null,
          },
        },
        400
      );
    }

    const rankedResults: RankedResult[] =
      await modelContext.rankingContext.rankAndSort(query, documents);

    return c.json({
      object: "list",
      model,
      data: rankedResults.map((result: RankedResult, index: number) => ({
        object: "rerank_result",
        document: result.document,
        relevance_score: result.score,
        index,
      })),
      usage: {
        prompt_tokens: -1, // Not available with local models
        total_tokens: -1,
      },
    });
  } catch (error: unknown) {
    console.error("Error reranking documents:", error);
    if (
      error instanceof Error &&
      "code" in error &&
      (error as any).code === "ENOENT"
    ) {
      return c.json(
        {
          error: {
            message: `Model '${
              (error as any).path
            }' not found in models directory`,
            type: "invalid_request_error",
            param: "model",
            code: "model_not_found",
          },
        },
        404
      );
    }
    return c.json(
      {
        error: {
          message:
            error instanceof Error ? error.message : "Unknown error occurred",
          type: "server_error",
          param: null,
          code: null,
        },
      },
      500
    );
  }
}
