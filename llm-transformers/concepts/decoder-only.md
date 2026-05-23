---
title: Decoder-only Models
status: todo
date: 2026-05-22
tags: [concept, decoder, gpt, generation]
---

# Decoder-only Models

Autoregressive models that generate text one token at a time. Each token can only attend to previous tokens (causal attention).

---

## What They Do

Given a prefix, predict the next token. Repeat to generate arbitrary-length text.

## Canonical Examples

GPT-2, GPT-3, GPT-4, LLaMA, Mistral, Gemini

## Good For

- Text generation / completion
- Chat / instruction following
- Code generation
- Few-shot prompting

## Not Good For

- Tasks that benefit from full bidirectional context (use encoder-only instead)

---

## Notes

<!-- fill in as the course progresses -->
