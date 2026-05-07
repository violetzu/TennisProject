DC = docker compose

up:
	$(DC) up -d

up-llm:
	$(DC) up -d vllm

up-embedding:
	$(DC) up -d vllm-embedding

up-ai:
	$(DC) --profile vllm up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

.PHONY: up up-llm up-embedding up-ai down logs
