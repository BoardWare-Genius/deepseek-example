## 需求
需要先安裝nvidia顯卡驅動

## python venv
先創建一個venv環境
```base
python3 -m venv-deepseek
```

```bash
pip install -r requirements.txt
```

```bash
docker run --gpus "device=0" <image name>
```
使用軟鏈結或直接將模型放到當前目錄的./model/目錄下，建議使用軟鏈結
```bash
ln -s /media/administrator/hdd/Workspace/Models/deepseek-model-8b ./model
```

## 使用Docker運行

```bash

```