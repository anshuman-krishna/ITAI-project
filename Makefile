.PHONY: help up down logs build ps clean

help:
	@echo "targets:"
	@echo "  up      start the full stack (detached)"
	@echo "  down    stop the stack"
	@echo "  logs    tail logs for all services"
	@echo "  build   rebuild all images"
	@echo "  ps      list running services"
	@echo "  clean   stop and remove volumes"

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

build:
	docker compose build

ps:
	docker compose ps

clean:
	docker compose down -v
