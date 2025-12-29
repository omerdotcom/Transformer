# Decoder-Only Transformer (from Scratch)

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Learning%20Project-yellow)
![Inspired By](https://img.shields.io/badge/Inspired%20By-Andrej%20Karpathy-orange)

## Overview
This project is a **decoder-only reimplementation of the Transformer architecture** built **from scratch in PyTorch**.

It follows **Andrej Karpathy’s NanoGPT / “Let’s build GPT” lecture series** step by step, with the goal of understanding how GPT-style models work at a low level. This is a **learning-focused implementation**, not a production-ready system.

## Features (So Far)
- Token embeddings  
- Positional embeddings  
<<<<<<< HEAD
- Multi-head Flash-attention
- Feed-forward networks  
- RMS normalization  
- Autoregressive text generation  
- Basic training loop  

This implementation currently matches the early/core parts of Karpathy’s walkthrough. More advanced features shown later have **not been added yet**.

## Motivation
The goal of this project is to:
- Learn how Transformer models are built from first principles  
- Understand GPT-style architectures without heavy abstractions  
- Practice implementing deep learning models directly in PyTorch  

This repository exists primarily as a **personal learning exercise**.

## Disclaimer
This code is **heavily inspired by and based on**  
**Andrej Karpathy’s NanoGPT / “Let’s build GPT” lecture series**.

The architecture, ideas, and overall structure come from Karpathy’s work.  
If you are looking for a full-featured or optimized implementation, you should use the **official NanoGPT repository** instead.
