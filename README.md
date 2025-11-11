# Confusion Matrix Explorer (PyShiny App)

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://denisecase.github.io/confusion-matrix-explorer/)
[![CI](https://github.com/denisecase/confusion-matrix-explorer/actions/workflows/ci.yml/badge.svg)](https://github.com/denisecase/confusion-matrix-explorer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

> This repo (`confusion-matrix-explorer`) contains a PyShiny app for exploring how changing a decision threshold affects a binary classifier confusion matrix and related metrics (sensitivity, specificity, precision, etc.).

## Launch the App In Your Browser

Click here: [**Launch the Confusion Matrix Explorer**](./docs/app/index.html)

The app runs in your browser using **Shinylive** (Pyodide), no installation needed.


## About the App

The **Confusion Matrix Explorer** app demonstrates how changing the decision threshold (the vertical T line) affects the confusion matrix and related metrics (sensitivity, specificity, etc.) for a binary classification problem.

How to use:

- Use the sidebar upper slider to vary the decision threshold.
- Use the sidebar lower slider to vary the number of bins in the histogram.
- Compare the as you raise (or lower) the decision threshold.

To learn more:

- See [ABOUT.md](./docs/ABOUT.md).

## Optional: Run on Your Machine

- To run on your machine, see the [Home Page](index.md).
- To make modifications, see [DEVELOPER.md](./docs/DEVELOPER.md)

---

## Screenshot (Raise the Bar)

![Raise the bar](./docs/images/bar_up.png)

## Screenshot (Default)

![Default](./docs/images/bar_center.png)

## Screenshot (Lower the Bar)

![Lower the bar](./docs/images/bar_down.png)

## License

This project is licensed under the [MIT License](LICENSE).
