# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.8-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    build-essential pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1