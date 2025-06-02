#!make
include .env
current_dir := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

## to build only one image use: make docker-build ARGS="comfyui"
docker-build:
	env USER_ID=${USER_ID} env GROUP_ID=${GROUP_ID} env COMPOSE_DOCKER_CLI_BUILD=1 env DOCKER_BUILDKIT=1 env COMPOSE_BAKE=true docker-compose build $(ARGS)

docker-rm:
	docker stop $(ARGS) && docker rm $(ARGS) && docker rmi $(ARGS)

docker-up:
	env USER_ID=${USER_ID} env GROUP_ID=${GROUP_ID} docker-compose up -d

docker-down:
	docker-compose down

#UTILITIES

get-nodes:
	bash bash/get-nodes.sh

docker-export:
	bash bash/docker-export.sh

dev-ssl:
	bash bash/dev-ssl.sh