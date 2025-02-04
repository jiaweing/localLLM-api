import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..");

// Default models directories
export const MODELS_BASE_DIR = path.join(projectRoot, "models");
export const MODEL_DIRS = {
  embedding: path.join(MODELS_BASE_DIR, "embedding"),
  reranker: path.join(MODELS_BASE_DIR, "reranker"),
  chat: path.join(MODELS_BASE_DIR, "chat"),
} as const;

export const PORT = 23673;
