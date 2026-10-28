#!/usr/bin/env bash
set -e

echo "=== [BigCodeBench] Installing System Utilities on RunPod ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        git \
        git-lfs \
        tmux \
        curl \
        ca-certificates \
        build-essential
fi

echo "=== Installing Python Core Sandbox Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing BigCodeBench Evaluation Requirements ==="
pip install \
    seaborn \
    matplotlib \
    scipy \
    scikit-learn \
    statsmodels \
    sympy \
    nltk \
    folium \
    geopy \
    geopandas \
    Django \
    flask \
    flask_login \
    flask_restful \
    WTForms \
    openpyxl \
    xlrd \
    xlwt \
    Pillow \
    beautifulsoup4 \
    lxml \
    xmltodict \
    cryptography \
    pycryptodome \
    Faker \
    python-docx \
    wordcloud \
    textblob \
    wikipedia \
    soundfile \
    requests-mock \
    pytest-timeout \
    psutil \
    tabulate \
    prettytable \
    requests \
    tqdm \
    rich

echo "=== Downloading Required NLTK Corpora ==="
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('averaged_perceptron_tagger', quiet=True)"

echo "=== BigCodeBench Setup Complete! ==="
echo "To run BigCodeBench Qwen2.5-Coder-7B data collection:"
echo "  python scripts/collect_bigcodebench.py --model Qwen/Qwen2.5-Coder-7B-Instruct --subset full --split instruct --n_rollouts 3 --temperature 0.7 --max_steps 5 --timeout 15.0 --output_dir artifacts/bigcodebench/qwen7b_full_k3"
