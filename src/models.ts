import fs from "fs/promises";
import { getLlama } from "node-llama-cpp";
import path from "path";
import { MODEL_DIRS } from "./config";
import type { ModelContext, ModelType } from "./types";

// Initialize llama.cpp
const llama = await getLlama();

// Map to store loaded models and their contexts
const loadedModels = new Map<string, ModelContext>();

// Create model directories if they don't exist
export async function initializeModelDirectories() {
  await fs.mkdir(path.dirname(MODEL_DIRS.chat), { recursive: true });
  await Promise.all(
    Object.values(MODEL_DIRS).map((dir) => fs.mkdir(dir, { recursive: true }))
  );
}

// Function to load a model and create appropriate context
export async function loadModel(
  modelName: string,
  type: ModelType
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
  }

  loadedModels.set(modelPath, context);
  return context;
}

// Function to unload unused models
export async function unloadUnusedModels(maxAgeMs: number = 30 * 60 * 1000) {
  const now = Date.now();
  for (const [path, context] of loadedModels.entries()) {
    if (now - context.lastUsed > maxAgeMs) {
      await context.model.dispose();
      loadedModels.delete(path);
    }
  }
}

// Function to unload a specific model
export async function unloadModel(modelName: string): Promise<boolean> {
  // Find the model in one of the model directories
  let modelPath: string | undefined;
  for (const dir of Object.values(MODEL_DIRS)) {
    const testPath = path.join(dir, modelName + ".gguf");
    if (loadedModels.has(testPath)) {
      modelPath = testPath;
      break;
    }
  }

  if (!modelPath) {
    return false;
  }

  const modelContext = loadedModels.get(modelPath)!;
  await modelContext.model.dispose();
  loadedModels.delete(modelPath);
  return true;
}

// Function to list all available models
export async function listModels() {
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

  return availableModels.flat();
}

// Setup automatic model unloading every 15 minutes
setInterval(() => {
  unloadUnusedModels().catch(console.error);
}, 15 * 60 * 1000);
