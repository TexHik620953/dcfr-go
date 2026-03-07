FROM golang:1.24 AS builder

ENV CGO_ENABLED=0
ENV GOOS=linux

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -ldflags="-s -w" -o /app/exec ./main.go

FROM alpine:3.19

WORKDIR /app

COPY --from=builder /app/exec /app/exec

CMD ["./exec"]
