.PHONY: up down logs dev-up dev-down dev-logs db-backup db-restore up-vllm setup help

DC      = docker compose
DC_DEV  = docker compose -f docker-compose.yml -f docker-compose.dev.yml

up:           ## Start production stack
	$(DC) up -d --build

down:         ## Stop all containers
	$(DC) down

logs:         ## Tail logs of all containers
	$(DC) logs -f

dev-up:       ## Start dev stack (frontend HMR + backend --reload)
	$(DC_DEV) up -d --build

dev-down:     ## Stop dev stack
	$(DC_DEV) down

dev-logs:     ## Tail all dev logs
	$(DC_DEV) logs -f

db-backup:    ## Backup database to backup.dump
	$(DC) exec -T db pg_dump -U tennis -Fc tennis > backup.dump

db-restore:   ## Restore database from backup.dump
	$(DC) exec -T db pg_restore -U tennis -d tennis --clean --if-exists < backup.dump

up-vllm:      ## Start production stack with vLLM profile
	$(DC) --profile vllm up -d --build

down-vllm:    ## Stop production stack with vLLM profile
	$(DC) --profile vllm down

up-embedding: ## Start production stack with embedding profile
	$(DC) --profile embedding up -d --build

down-embedding: ## Stop production stack with embedding profile
	$(DC) --profile embedding down

up-llm:       ## Start production stack with llm profile
	$(DC) --profile llm up -d --build	

down-llm:     ## Stop production stack with llm profile
	$(DC) --profile llm down

setup:        ## First-time setup: copy .env
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env — please fill in secrets before starting"; else echo ".env already exists, skipped"; fi

help:         ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
