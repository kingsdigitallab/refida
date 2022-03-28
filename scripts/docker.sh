docker build -t refida --target dev . && docker run --rm -p 8000:8000 -v $(pwd)/.:/app refida
