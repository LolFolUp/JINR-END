PROTOC      ?= protoc
PROTOC_GEN_GO ?= $(shell go env GOPATH)/bin/protoc-gen-go
GO          ?= go
MODULE      = github.com/aladed/JINR/edge-agent

.PHONY: proto build lint test clean

proto:
	$(PROTOC) --go_out=. \
		--go_opt=module=$(MODULE) \
		--plugin=protoc-gen-go=$(PROTOC_GEN_GO) \
		internal/transport/telemetry.proto

build:
	$(GO) build ./...

lint:
	$(GO) vet ./...

test:
	$(GO) test ./...

clean:
	$(GO) clean ./...
