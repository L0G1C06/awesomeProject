#!/bin/sh
# Inicia o servidor Ollama em background e puxa os modelos configurados

set -e

LLM_MODEL="${OLLAMA_LLM_MODEL:-llama3.2}"
EMBED_MODEL="${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

echo "[ollama] Iniciando servidor..."
ollama serve &
SERVER_PID=$!

# Aguarda o servidor estar pronto
echo "[ollama] Aguardando servidor iniciar..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo "[ollama] Servidor pronto!"

# Pull dos modelos se não existirem
for MODEL in "$LLM_MODEL" "$EMBED_MODEL"; do
  if ! ollama list | grep -q "$MODEL"; then
    echo "[ollama] Baixando modelo: $MODEL"
    ollama pull "$MODEL"
  else
    echo "[ollama] Modelo já disponível: $MODEL"
  fi
done

echo "[ollama] Todos os modelos prontos!"
wait $SERVER_PID