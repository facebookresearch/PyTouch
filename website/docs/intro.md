---
id: intro
title: What is PyTouch?
sidebar_label: About
slug: /
---

## Introduction

With the increased availability of rich tactile sensors, the sense of touch is becoming a new and important sensor modality in robotics and machine learning. Sensing the world through touch open exciting new challenges and opportunities to measure, understand and interact with the world around us. However, the availability of ready-to-use touch processing software is extremely limited whichs results in a high entry bar for new practitioners that want to make use of tactile sensors, which are forced to implement their own touch processing routines.

PyTouch is an open-source library for touch processing that enables the machine learning and the robotics community to process raw touch data from tactile sensors through abstractions which focus on the experiment instead of the low level details of elementary concepts.

The software library modularizes a set of commonly used tactile-processing functions valuable for various down-stream tasks, such as tactile manipulation, slip detection, object recognition based on touch and other touch based tasks.

PyTouch aims to standardize the way touch based experiments are designed in reducing the amount of individual software developed for one off experiments by using the PyTouch library as a foundation which can be expanded upon for future experimental and research applications.

## Key Features

- Built with PyTorch
- Pre-trained models for touch tasks are served on-demand based on initialized tasks
- Extendable framework for introducing new tasks
- Reusable framework for reducing boilerplate code across touch experiments and applications

## In Beta

PyTouch is currently in beta. Backwards compatability may break with minor version changes until release `1.0.0`

## Versions

| PyTouch Release | Notes           | Python Versions |
| --------------- | --------------- | --------------- |
| **0.4.0**       | Initial Release | **3.7 - 3.9**   |
