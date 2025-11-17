## Usage

This project uses [**uv**](https://github.com/astral-sh/uv) for high-performance Python package management. Follow these steps to get your environment set up and run the code.

### 1\. Prerequisites

  * Python 3.12+
  * `uv` (see [installation guide](https://www.google.com/search?q=https://github.com/astral-sh/uv%23installation)):
    ```bash
    # Example (macOS/Linux):
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2\. Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/F1xedbot/PCCEntropy.git
    cd PCCEntropy
    ```

2.  **Create a virtual environment:**
    `uv` will create and manage a virtual environment in a `.venv` directory.

    ```bash
    uv venv
    ```

3.  **Activate the environment:**

      * **macOS / Linux:**
        ```bash
        source .venv/bin/activate
        ```
      * **Windows (PowerShell):**
        ```powershell
        .venv\Scripts\Activate.ps1
        ```

4.  **Sync dependencies:**
    Use `uv sync --active` to install all required dependencies.

    ```bash
    uv sync --active
    ```

### 3\. Running the Project

This project has several entry points depending on your goal.

**Note on `PYTHONPATH`:**
Before running scripts, you may need to add the project's root directory to your `PYTHONPATH` for imports to work correctly.

  * **Windows (PowerShell):**
    Run the provided setup script:
    ```powershell
    .\.local_env.ps1
    ```
  * **macOS / Linux:**
    Run this command in your terminal:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)
    ```

-----

#### A. Run the Full Pipeline (From Scratch)

To train all models and generate all data from scratch, run the scripts from the `src/scripts/` directory in this order:

1.  ```bash
    uv run python src/scripts/train_and_evaluate_base_model.py
    ```
2.  ```bash
    uv run python src/scripts/create_vectorstore.py
    ```
3.  ```bash
    uv run python src/scripts/run_counterfactual_corrector.py
    ```

-----

#### B. Run an Example Inference

To run a single example inference using the pre-trained models, execute `src/main.py`:

```bash
uv run python src/main.py
```

-----

#### C. Explore the Analysis Notebook

To view the data, explore the analysis, and see the result visualizations, you can use the Jupyter Notebook.

```bash
# Make sure your virtual environment is active
uv run jupyter notebook notebook/analysis.ipynb
```