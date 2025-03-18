FROM python:3.9-slim

WORKDIR /app

# OpenCVの依存関係をインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 必要なライブラリをインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# スクリプトをコピー
COPY src/ .

# ボリュームマウントポイントを作成
VOLUME ["/app/data", "/app/output"]

# コンテナ起動時のコマンド
ENTRYPOINT ["python", "analyzer.py"]
# デフォルト引数（オーバーライド可能）
CMD ["/app/data/input.mp4"]