build:
	chmod +x ./deployment/deploy.sh
	./deployment/deploy.sh build

up:
	./deployment/deploy.sh up

down:
	./deployment/deploy.sh down

lint:
	ruff check --fix ./src || true

format:
	ruff format ./src || true
