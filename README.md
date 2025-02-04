# Local LLM API Server

An API server that provides OpenAI-compatible endpoints for running GGUF models locally. Designed for easy integration with any system that supports OpenAI's API format.

## Overview

This service allows you to use your own GGUF models (like llama.cpp models) through an API that mimics OpenAI's interface. This makes it easy to:

- Replace OpenAI APIs with local alternatives in existing applications
- Run AI operations locally for better privacy and control
- Use the same code with both OpenAI and local models

Supported Operations:

- Chat Completions (like GPT-3.5/4)
- Embeddings Generation
- Document Reranking

## Features

- ✨ OpenAI-compatible API endpoints
- 🚀 Drop-in replacement for OpenAI's client libraries
- 🔒 Run models locally for privacy and cost savings
- 🔄 Auto-loading and unloading of models for memory efficiency
- 📁 Organized model management by type (chat/embedding/reranking)

## Setup

1. Clone and set up:

```bash
git clone https://github.com/jiaweing/localLLM-api.git
cd localLLM-api
pnpm install
pnpm build    # Builds to dist/ directory
```

2. Place your GGUF models in the appropriate directories under `models/`:

```
localLLM/
  ├── models/
  │   ├── embedding/          # Embedding models
  │   │   └── all-MiniLM-L6-v2.Q4_K_M.gguf
  │   ├── reranker/          # Reranking models
  │   │   └── bge-reranker-v2-m3-Q8_0.gguf
  │   └── chat/              # Chat completion models
  │       └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Note: The `.gguf` extension will be automatically appended if not provided in API requests.

3. Run the service:

Development mode (with auto-reload):

```bash
pnpm dev    # Runs TypeScript watch mode and starts server
```

Production mode:

```bash
pnpm start  # Starts server from compiled dist/
```

The service will start on port 23673.

Note: The service automatically creates the required `models/` subdirectories on startup.

## API Endpoints

### Chat Completions

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

```json
{
  "model": "Llama-3.2-1B-Instruct-Q4_K_M",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello! Can you help me?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1000
}
```

Response:

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1677649420,
  "model": "Llama-3.2-1B-Instruct-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Of course! I'd be happy to help. What can I assist you with?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": -1,
    "completion_tokens": -1,
    "total_tokens": -1
  }
}
```

### Embedding Generation

#### `POST /v1/embeddings`

Generate embeddings for text inputs. OpenAI-compatible endpoint format.

```json
{
  "model": "all-MiniLM-L6-v2.Q4_K_M",
  "input": "Your text to embed"
  // Or array of strings: ["text1", "text2", ...]
}
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [
        /* vector of numbers */
      ],
      "index": 0
    }
    // More embeddings if input was an array
  ],
  "model": "all-MiniLM-L6-v2.Q4_K_M",
  "usage": {
    "prompt_tokens": -1,
    "total_tokens": -1
  }
}
```

### Document Reranking

#### `POST /v1/rerank`

Rerank a list of documents based on relevance to a query. OpenAI-style endpoint format.

```json
{
  "model": "bge-reranker-v2-m3-Q8_0",
  "query": "Your search query",
  "documents": ["doc1", "doc2", "doc3"]
}
```

Response:

```json
{
  "object": "list",
  "model": "bge-reranker-v2-m3-Q8_0",
  "data": [
    {
      "object": "rerank_result",
      "document": "Most relevant document",
      "relevance_score": 0.95,
      "index": 0
    }
    // ... more documents in descending order of relevance
  ],
  "usage": {
    "prompt_tokens": -1,
    "total_tokens": -1
  }
}
```

### Model Management

#### `POST /v1/models/load`

Pre-load a model into memory.

```json
{
  "model": "Llama-3.2-1B-Instruct-Q4_K_M",
  "type": "chat" // or "embedding" or "reranker"
}
```

#### `POST /v1/models/unload`

Unload a model from memory.

```json
{
  "model": "Llama-3.2-1B-Instruct-Q4_K_M"
}
```

#### `GET /v1/models`

List all available models. Response includes:

- Name (without .gguf extension)
- Type (embedding, reranker, or chat)
- Load status (whether model is currently loaded)

Example response:

```json
[
  {
    "name": "all-MiniLM-L6-v2.Q4_K_M",
    "type": "embedding",
    "loaded": true
  },
  {
    "name": "bge-reranker-v2-m3-Q8_0",
    "type": "reranker",
    "loaded": false
  },
  {
    "name": "Llama-3.2-1B-Instruct-Q4_K_M",
    "type": "chat",
    "loaded": true
  }
]
```

## Error Handling

All endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad Request (missing/invalid parameters)
- 404: Not Found (model not found in appropriate directory)
- 500: Internal Server Error

Error responses follow OpenAI's format:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

## Example Usage

### With OpenAI's Node.js Library

```typescript
import OpenAI from "openai";

const openai = new OpenAI({
  baseURL: "http://localhost:23673/v1", // Point to local server
  apiKey: "not-needed", // API key is not required but must be non-empty
});

// Chat completions
const chatCompletion = await openai.chat.completions.create({
  model: "Llama-3.2-1B-Instruct-Q4_K_M",
  messages: [{ role: "user", content: "Hello!" }],
});

// Embeddings
const embedding = await openai.embeddings.create({
  model: "all-MiniLM-L6-v2.Q4_K_M",
  input: "Hello world",
});
```

### With cURL

```bash
# Chat Completions
curl -X POST http://localhost:23673/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.2-1B-Instruct-Q4_K_M",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    "temperature": 0.7
  }'

# Embeddings
curl -X POST http://localhost:23673/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "all-MiniLM-L6-v2.Q4_K_M",
    "input": "Example text to embed"
  }'

# Reranking
curl -X POST http://localhost:23673/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3-Q8_0",
    "query": "search query",
    "documents": [
      "First document to rank",
      "Second document to rank",
      "Third document to rank"
    ]
  }'
```

## Memory Management

Models are automatically unloaded after 30 minutes of inactivity to manage memory usage. You can:

1. Preload models using `/models/load`
2. Check available models with `/models/list`
3. Manually unload models with `/models/unload`
