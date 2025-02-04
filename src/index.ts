import { serve } from "@hono/node-server";
import fs from "fs/promises";
import type { Context } from "hono";
import { Hono } from "hono";
import {
  getLlama,
  LlamaChatSession,
  TemplateChatWrapper,
} from "node-llama-cpp";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");

// Initialize llama.cpp
const llama = await getLlama();

// Default models directories (will create if they don't exist)
const MODELS_BASE_DIR = path.join(projectRoot, "models");
const MODEL_DIRS = {
  embedding: path.join(MODELS_BASE_DIR, "embedding"),
  reranker: path.join(MODELS_BASE_DIR, "reranker"),
  chat: path.join(MODELS_BASE_DIR, "chat"),
};

// Create model directories if they don't exist
await fs.mkdir(MODELS_BASE_DIR, { recursive: true });
await Promise.all(
  Object.values(MODEL_DIRS).map((dir) => fs.mkdir(dir, { recursive: true }))
);

// Map to store loaded models and their contexts
interface ModelContext {
  model: any;
  embeddingContext?: any;
  rankingContext?: any;
  lastUsed: number;
}

const loadedModels = new Map<string, ModelContext>();

// Function to load a model and create appropriate context
async function loadModel(
  modelName: string,
  type: "embedding" | "reranker" | "chat"
): Promise<ModelContext> {
  // Append .gguf extension if not present
  const modelFileName = modelName.endsWith(".gguf")
    ? modelName
    : `${modelName}.gguf`;
  const modelPath = path.join(MODEL_DIRS[type], modelFileName);
  const existingModel = loadedModels.get(modelPath);
  if (existingModel) {
    existingModel.lastUsed = Date.now();
    return existingModel;
  }

  const model = await llama.loadModel({
    modelPath,
  });

  const context: ModelContext = {
    model,
    lastUsed: Date.now(),
  };

  if (type === "embedding") {
    context.embeddingContext = await model.createEmbeddingContext();
  } else if (type === "reranker") {
    context.rankingContext = await model.createRankingContext();
  } else if (type === "chat") {
    context.model = model; // No special context needed for chat completion
  }

  loadedModels.set(modelPath, context);
  return context;
}

// Function to unload unused models (called periodically)
async function unloadUnusedModels(maxAgeMs: number = 30 * 60 * 1000) {
  // 30 minutes default
  const now = Date.now();
  for (const [path, context] of loadedModels.entries()) {
    if (now - context.lastUsed > maxAgeMs) {
      await context.model.dispose();
      loadedModels.delete(path);
    }
  }
}

// Set up automatic model unloading every 15 minutes
setInterval(() => {
  unloadUnusedModels().catch(console.error);
}, 15 * 60 * 1000);

const app = new Hono();

app.get("/", (c: Context) => {
  return c.text("Local LLM Service");
});

// OpenAI-compatible embeddings endpoint
app.post("/v1/embeddings", async (c: Context) => {
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
});

// OpenAI-style reranking endpoint
app.post("/v1/rerank", async (c: Context) => {
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

    interface RankedResult {
      document: string;
      score: number;
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
});

// Model management endpoints
app.post("/v1/models/load", async (c: Context) => {
  try {
    const { model, type } = await c.req.json();

    if (!model || !type) {
      return c.json(
        { error: "Model name and type (embedding or reranker) are required" },
        400
      );
    }

    if (type !== "embedding" && type !== "reranker" && type !== "chat") {
      return c.json(
        { error: "Type must be either 'embedding', 'reranker', or 'chat'" },
        400
      );
    }

    await loadModel(model, type);
    return c.json({ message: "Model loaded successfully" });
  } catch (error: unknown) {
    console.error("Error loading model:", error);
    if (
      error instanceof Error &&
      "code" in error &&
      (error as any).code === "ENOENT"
    ) {
      return c.json(
        {
          error: `Model '${(error as any).path}' not found in models directory`,
        },
        404
      );
    }
    return c.json(
      {
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

app.post("/v1/models/unload", async (c: Context) => {
  try {
    const { model } = await c.req.json();

    if (!model) {
      return c.json({ error: "Model name is required" }, 400);
    }

    // Find the model in one of the model directories
    let modelPath: string | undefined;
    for (const dir of Object.values(MODEL_DIRS)) {
      const testPath = path.join(dir, model + ".gguf");
      if (loadedModels.has(testPath)) {
        modelPath = testPath;
        break;
      }
    }

    if (!modelPath) {
      return c.json({ error: "Model not found or not loaded" }, 404);
    }
    const modelContext = loadedModels.get(modelPath)!; // We know it exists because we checked with loadedModels.has()
    await modelContext?.model.dispose();
    loadedModels.delete(modelPath);
    return c.json({ message: "Model unloaded successfully" });
  } catch (error) {
    console.error("Error unloading model:", error);
    return c.json(
      {
        error:
          error instanceof Error ? error.message : "Unknown error occurred",
      },
      500
    );
  }
});

app.get("/v1/models", async (c: Context) => {
  try {
    // Get available models from each directory
    const availableModels = await Promise.all(
      Object.entries(MODEL_DIRS).map(async ([type, dir]) => {
        try {
          const files = await fs.readdir(dir);
          return files
            .filter((file) => file.endsWith(".gguf"))
            .map((file) => ({
              name: path.basename(file, ".gguf"),
              type,
              loaded: loadedModels.has(path.join(dir, file)),
            }));
        } catch (error) {
          console.warn(`Error reading ${type} models directory:`, error);
          return [];
        }
      })
    );

    // Flatten the array of arrays
    const models = availableModels.flat();

    return c.json(models);
  } catch (error) {
    console.error("Error listing models:", error);
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
});

// OpenAI-compatible chat completion endpoint
app.post("/v1/chat/completions", async (c: Context) => {
  try {
    const {
      model,
      messages,
      temperature = 0.7,
      max_tokens,
      stream = false,
    } = await c.req.json();

    if (!model || !Array.isArray(messages) || messages.length === 0) {
      return c.json(
        {
          error: {
            message: "Missing required parameters: model, messages (array)",
            type: "invalid_request_error",
            param: !model ? "model" : "messages",
            code: null,
          },
        },
        400
      );
    }

    // Create a template chat wrapper for consistent formatting
    const chatWrapper = new TemplateChatWrapper({
      template:
        "{{systemPrompt}}\n{{history}}Assistant: {{completion}}\nHuman: ",
      historyTemplate: {
        system: "System: {{message}}\n",
        user: "Human: {{message}}\n",
        model: "Assistant: {{message}}\n",
      },
    });

    const chatSessions = new Map<string, LlamaChatSession>();

    const modelContext = await loadModel(model, "chat");
    const sessionId = `${model}-${Date.now()}`;

    // Process messages
    let systemPrompt = "";
    for (const msg of messages) {
      if (msg.role === "system") {
        systemPrompt = msg.content;
      }
    }

    // Create or get existing chat session
    let session = chatSessions.get(sessionId);
    if (!session) {
      // Create a new context and session
      const context = await modelContext.model.createContext();
      session = new LlamaChatSession({
        systemPrompt,
        contextSequence: context.getSequence(),
        chatWrapper,
      });
      chatSessions.set(sessionId, session);

      // Clean up old sessions periodically
      setTimeout(() => {
        chatSessions.delete(sessionId);
      }, 30 * 60 * 1000); // 30 minutes
    }

    if (stream) {
      // Streaming response
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          try {
            let buffer = "";
            const response = await session!.prompt(
              messages[messages.length - 1].content,
              {
                temperature,
                maxTokens: max_tokens,
                onTextChunk(token: string) {
                  buffer += token;
                  if (buffer.includes(" ") || buffer.includes("\n")) {
                    const chunk = {
                      id: sessionId,
                      object: "chat.completion.chunk",
                      created: Date.now(),
                      model,
                      choices: [
                        {
                          index: 0,
                          delta: {
                            content: buffer,
                          },
                          finish_reason: null,
                        },
                      ],
                    };
                    controller.enqueue(
                      encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                    );
                    buffer = "";
                  }
                },
              }
            );

            // Send any remaining buffer
            if (buffer) {
              const chunk = {
                id: sessionId,
                object: "chat.completion.chunk",
                created: Date.now(),
                model,
                choices: [
                  {
                    index: 0,
                    delta: {
                      content: buffer,
                    },
                    finish_reason: "stop",
                  },
                ],
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
              );
            }

            // Send [DONE] message
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        },
      });

      return new Response(stream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      // Non-streaming response
      const response = await session.prompt(
        messages[messages.length - 1].content,
        {
          temperature,
          maxTokens: max_tokens,
        }
      );

      return c.json({
        id: sessionId,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: response,
            },
            finish_reason: "stop",
          },
        ],
        usage: {
          prompt_tokens: -1,
          completion_tokens: -1,
          total_tokens: -1,
        },
      });
    }
  } catch (error: unknown) {
    console.error("Error generating chat completion:", error);
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
});

const port = 23673;
console.log(`Local LLM Service is running on http://localhost:${port}`);

serve({
  fetch: app.fetch,
  port,
});
