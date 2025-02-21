## 需求
需要先安裝nvidia顯卡驅動

## python venv
先創建一個venv環境
```base
python3 -m venv-deepseek
```

```bash
pip install torch
# 這個安裝包非常大
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

```bash
docker run --gpus "device=0" <image name>
```

```bash
helm repo add rancher-latest https://releases.rancher.com/server-charts/latest

kubectl create namespace cattle-system

kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.16/cert-manager.crds.yaml


helm repo add jetstack https://charts.jetstack.io

helm repo update

helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace --kubeconfig /etc/rancher/k3s/k3s.yaml

helm install rancher rancher-latest/rancher \
  --namespace cattle-system \
  --set hostname=rancher.k3s.bw \
  --set replicas=1 \
  --set bootstrapPassword=<change me> \
  --kubeconfig /etc/rancher/k3s/k3s.yaml 
```