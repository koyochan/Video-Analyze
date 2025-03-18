import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go

def extract_people_from_video(video_path, output_csv="people_tracks.csv", sample_rate=1):
    """
    MP4動画から人物を検出し、追跡データをCSVファイルに保存する
    
    パラメータ:
    - video_path: 動画ファイルのパス
    - output_csv: 出力CSVファイルのパス
    - sample_rate: 何フレームに1回処理するか（処理を軽くするため）
    """
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    
    # 動画が正常に開けたか確認
    if not cap.isOpened():
        print(f"エラー: {video_path} を開けませんでした。")
        return None
    
    # 動画の情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"動画情報: {width}x{height}, {fps}fps, 総フレーム数: {frame_count}")
    
    # HOG人物検出器の初期化
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # 追跡用のオブジェクト（OpenCVバージョンに依存しないようにディクショナリで管理）
    trackers = []
    track_id = 0
    
    # 追跡データを保存するリスト
    tracking_data = []
    
    # 現在のフレーム番号
    frame_number = 0
    
    # 動画の開始時刻（現在時刻から動画の長さを引いた時間）
    video_duration_seconds = frame_count / fps
    start_time = datetime.now() - timedelta(seconds=video_duration_seconds)
    
    # プログレスバーの設定
    pbar = tqdm(total=frame_count)

    # OpenCVバージョンの確認とトラッカー作成方法の決定
    opencv_version = cv2.__version__.split('.')
    major_version = int(opencv_version[0])
    
    # 前のフレームで検出された人物のBoundingBox
    prev_boxes = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        pbar.update(1)
        
        # sample_rateフレームごとに処理
        if frame_number % sample_rate != 0:
            continue
        
        # 現在のフレームの時刻
        current_time = start_time + timedelta(seconds=(frame_number / fps))
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        
        # フレームのリサイズ（処理を高速化するため）
        frame_resized = cv2.resize(frame, (width // 2, height // 2))
        
        # 10フレームごとに人物を検出
        if frame_number % (10 * sample_rate) == 0:
            # 人物検出
            boxes, weights = hog.detectMultiScale(
                frame_resized, 
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.05
            )
            
            # リサイズしたフレームでの座標を元のサイズに戻す
            detected_boxes = []
            for box in boxes:
                x, y, w, h = [v * 2 for v in box]
                detected_boxes.append((x, y, w, h))
            
            # 検出された人物を追跡データとして記録
            for i, (x, y, w, h) in enumerate(detected_boxes):
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 新しいトラックIDを割り当て
                current_track_id = f"person_{track_id + i}"
                
                # トラッキングデータを保存
                tracking_data.append({
                    'track_id': current_track_id,
                    'frame': frame_number,
                    'timestamp': timestamp,
                    'x': center_x,
                    'y': center_y,
                    'width': width,
                    'height': height
                })
            
            # 次のフレーム用にtrack_idを更新
            track_id += len(detected_boxes)
            
            # 次のフレームの検出用に現在のボックスを保存
            prev_boxes = detected_boxes
        
        # 検出の間のフレームでは、前回検出された位置の周辺を簡易的に探索
        elif prev_boxes:
            # 簡易的な追跡: 前回のボックスの周辺でテンプレートマッチングや特徴点追跡を行うなど
            # ここではとりあえず前回の位置をそのまま使用
            for i, (x, y, w, h) in enumerate(prev_boxes):
                if frame_number % (2 * sample_rate) == 0:  # 間引いて記録
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    tracking_data.append({
                        'track_id': f"person_{track_id - len(prev_boxes) + i}",
                        'frame': frame_number,
                        'timestamp': timestamp,
                        'x': center_x,
                        'y': center_y,
                        'width': width,
                        'height': height
                    })
    
    pbar.close()
    cap.release()
    
    # トラッキングデータをDataFrameに変換
    if tracking_data:
        df = pd.DataFrame(tracking_data)
        df.to_csv(output_csv, index=False)
        print(f"トラッキングデータを {output_csv} に保存しました")
        return df
    else:
        print("トラッキングデータが取得できませんでした")
        return None

def create_heatmap_from_pixel_data(df, output_file="heatmap.html"):
    """
    ピクセルベースのトラッキングデータからヒートマップを作成する
    
    パラメータ:
    - df: トラッキングデータを含むDataFrame（x, y列が必要）
    - output_file: 出力HTMLファイルのパス
    """
    # 2Dヒストグラムの作成
    fig = px.density_heatmap(
        df, 
        x="x", 
        y="y", 
        nbinsx=50,  # 必要に応じてビン数を調整
        nbinsy=50,
        histfunc="count",
        labels={"x": "X座標", "y": "Y座標", "color": "カウント"},
        title="人物密度ヒートマップ（ピクセル座標）"
    )
    
    # Y軸を反転（画像座標では0,0が左上のため）
    fig.update_yaxes(autorange="reversed")
    
    # ヒートマップの保存
    fig.write_html(output_file)
    print(f"ヒートマップを {output_file} に保存しました")
    
    return fig

def create_flow_visualization(df, output_file="flow_map.html"):
    """
    ピクセルベースのトラッキングデータから流れの可視化を作成する
    
    パラメータ:
    - df: トラッキングデータを含むDataFrame（track_id, x, y, timestamp列が必要）
    - output_file: 出力HTMLファイルのパス
    """
    # 図の作成
    fig = go.Figure()
    
    # track_idごとに処理
    for track_id, track_df in df.groupby('track_id'):
        # タイムスタンプでソート
        track_df = track_df.sort_values('timestamp')
        
        if len(track_df) >= 2:  # 少なくとも2点あるときのみ描画
            # このトラックの線トレースを追加
            fig.add_trace(go.Scatter(
                x=track_df['x'],
                y=track_df['y'],
                mode='lines+markers',
                name=track_id,
                line=dict(width=2),
                marker=dict(size=5)
            ))
    
    # レイアウトの更新
    fig.update_layout(
        title="人物の移動フロー",
        xaxis_title="X座標",
        yaxis_title="Y座標",
        yaxis=dict(autorange="reversed"),  # Y軸を反転
        legend_title="人物ID"
    )
    
    # フローマップの保存
    fig.write_html(output_file)
    print(f"フロー可視化を {output_file} に保存しました")
    
    return fig

def create_dashboard(df, output_file="dashboard.html"):
    """
    トラッキングデータからインタラクティブなダッシュボードを作成する
    
    パラメータ:
    - df: トラッキングデータを含むDataFrame
    - output_file: 出力HTMLファイルのパス
    """
    # タイムスタンプから時間情報を抽出
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    
    # 10分間隔でグループ化
    df['time_bin'] = df['hour'] * 6 + df['minute'] // 10
    time_counts = df.groupby('time_bin').size().reset_index(name='count')
    time_counts['time_label'] = time_counts['time_bin'].apply(
        lambda x: f"{x // 6:02d}:{(x % 6) * 10:02d}"
    )
    
    # 時間ごとの人数グラフ
    fig1 = px.bar(
        time_counts, 
        x='time_label', 
        y='count', 
        title='10分間隔での人数',
        labels={'time_label': '時間', 'count': '人数'}
    )
    
    # ピクセルデータのグリッドベースカウント作成
    df['x_grid'] = (df['x'] / 20).astype(int) * 20  # 20ピクセルグリッド
    df['y_grid'] = (df['y'] / 20).astype(int) * 20
    area_counts = df.groupby(['x_grid', 'y_grid']).size().reset_index(name='count')
    
    # エリア別のヒートマップ
    fig2 = px.density_heatmap(
        area_counts,
        x='x_grid', 
        y='y_grid', 
        z='count',
        nbinsx=50,
        nbinsy=50,
        title='エリア別の密度'
    )
    fig2.update_yaxes(autorange="reversed")  # Y軸を反転
    
    # ユニークなトラックの時系列
    track_time = df.groupby(['time_bin', 'track_id']).size().reset_index(name='points')
    unique_tracks_per_time = track_time.groupby('time_bin').size().reset_index(name='unique_tracks')
    unique_tracks_per_time['time_label'] = unique_tracks_per_time['time_bin'].apply(
        lambda x: f"{x // 6:02d}:{(x % 6) * 10:02d}"
    )
    
    fig3 = px.line(
        unique_tracks_per_time, 
        x='time_label', 
        y='unique_tracks', 
        title='時間ごとの人数推移',
        labels={'time_label': '時間', 'unique_tracks': '人数'}
    )
    
    # HTMLファイルとして保存
    with open(output_file, 'w') as f:
        f.write('<html><head><title>人流分析ダッシュボード</title></head><body>\n')
        f.write('<h1>人流分析ダッシュボード</h1>\n')
        f.write(fig3.to_html(full_html=False))
        f.write('<br><br>\n')
        f.write(fig1.to_html(full_html=False))
        f.write('<br><br>\n')
        f.write(fig2.to_html(full_html=False))
        f.write('</body></html>')
    
    print(f"ダッシュボードを {output_file} に保存しました")

def main(video_path, output_dir="output", sample_rate=5):
    """
    メイン処理：MP4動画からの人流分析と可視化を行う
    
    パラメータ:
    - video_path: 動画ファイルのパス
    - output_dir: 出力ディレクトリ
    - sample_rate: 何フレームに1回処理するか
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイルパスの設定
    tracks_csv = os.path.join(output_dir, "people_tracks.csv")
    heatmap_file = os.path.join(output_dir, "people_heatmap.html")
    flow_map_file = os.path.join(output_dir, "people_flow_map.html")
    dashboard_file = os.path.join(output_dir, "people_dashboard.html")
    
    # 動画からのトラッキング（既に処理済みの場合はスキップ）
    if os.path.exists(tracks_csv):
        print(f"{tracks_csv} が既に存在します。読み込みます...")
        df = pd.read_csv(tracks_csv)
    else:
        print(f"{video_path} から人物を検出・追跡しています...")
        df = extract_people_from_video(video_path, tracks_csv, sample_rate)
    
    if df is None or len(df) == 0:
        print("トラッキングデータが取得できませんでした。処理を終了します。")
        return
    
    # 各種可視化の作成
    print("ヒートマップを作成しています...")
    create_heatmap_from_pixel_data(df, heatmap_file)
    
    print("フロー可視化を作成しています...")
    create_flow_visualization(df, flow_map_file)
    
    print("ダッシュボードを作成しています...")
    create_dashboard(df, dashboard_file)
    
    print("分析が完了しました。以下のファイルが作成されました:")
    print(f" - トラッキングデータ: {tracks_csv}")
    print(f" - ヒートマップ: {heatmap_file}")
    print(f" - フロー可視化: {flow_map_file}")
    print(f" - ダッシュボード: {dashboard_file}")

# 使用例
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        main(video_path)
    else:
        print("使用方法: python script.py <video_path>")
        print("例: python script.py input.mp4")