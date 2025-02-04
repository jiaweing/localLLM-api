export interface ModelContext {
  model: any;
  embeddingContext?: any;
  rankingContext?: any;
  lastUsed: number;
}

export interface RankedResult {
  document: string;
  score: number;
}

export type ModelType = "embedding" | "reranker" | "chat";
