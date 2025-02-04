import type { Context } from "hono";
import { loadModel } from "./models";

export async function handleEmbeddings(c: Context) {
  try {
    const { model, input } = await c.req.json();

    if (!model || !input) {
      return c.json(
        {
          error: {
            message: "Missing required parameters: model, input",
            type: "invalid_request_error",
            param: !model ? "model" : "input",
            code: null,
          },
        },
        400
      );
    }

    const modelContext = await loadModel(model, "embedding");
    if (!modelContext.embeddingContext) {
      return c.json(
        {
          error: {
            message: "Model is not suitable for embeddings",
            type: "invalid_request_error",
            param: "model",
            code: null,
          },
        },
        400
      );
    }

    // Handle both single string and array of strings
    const inputs = Array.isArray(input) ? input : [input];
    const embeddings = [];

    for (const text of inputs) {
      const embedding = await modelContext.embeddingContext.getEmbeddingFor(
        text
      );
      embeddings.push({
        embedding: [...embedding.vector],
        index: embeddings.length,
      });
    }

    return c.json({
      object: "list",
      data: embeddings.map(({ embedding, index }) => ({
        object: "embedding",
        embedding,
        index,
      })),
      model,
      usage: {
        prompt_tokens: -1, // Not available with local models
        total_tokens: -1,
      },
    });
  } catch (error: unknown) {
    console.error("Error generating embedding:", error);
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
