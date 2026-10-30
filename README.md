# RAG QA 챗봇

Streamlit 프론트엔드와 FastAPI 백엔드를 분리한 RAG QA 데모입니다. Python **3.11**을 사용하며, 임베딩/생성 구현체는 각각 `mock` 또는 HTTP API로 선택합니다. 벡터 검색은 현재 mock이며 pgvector 구현은 추후 추가할 예정입니다.

## 폴더 구조

```text
backend/       FastAPI API, 검색/생성 provider, pytest, requirements.txt
model_api/     FastAPI 임베딩 API, sentence-transformers, pytest
frontend/      Streamlit 채팅 UI, requirements.txt
deploy/        backend.service, model-api.service, frontend.service (systemd)
run_backend.sh / run_backend.bat
run_frontend.sh / run_frontend.bat
model_api/run_model_api.sh / .bat
.env.example
```

## 로컬 실행

### 1. Python 3.11 설치 및 환경 파일 생성

Python 3.11을 설치하고, 프로젝트 루트에서 환경 파일을 만듭니다.

```powershell
Copy-Item .env.example .env
```

### 2. 가상환경 생성 및 패키지 설치

Windows PowerShell:

```powershell
py -3.11 -m venv backend\.venv
py -3.11 -m venv frontend\.venv
py -3.11 -m venv model_api\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
frontend\.venv\Scripts\python.exe -m pip install -r frontend\requirements.txt
model_api\.venv\Scripts\python.exe -m pip install -r model_api\requirements.txt
```

Ubuntu/macOS:

```bash
python3.11 -m venv backend/.venv
python3.11 -m venv frontend/.venv
python3.11 -m venv model_api/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
frontend/.venv/bin/python -m pip install -r frontend/requirements.txt
model_api/.venv/bin/python -m pip install -r model_api/requirements.txt
```

### 3. 서버 실행

서로 다른 세 터미널에서 프로젝트 루트 기준으로 실행합니다.

Windows:

```powershell
.\run_backend.bat
.\run_frontend.bat
.\model_api\run_model_api.bat
```

Ubuntu/macOS:

```bash
chmod +x run_backend.sh run_frontend.sh
chmod +x model_api/run_model_api.sh
./run_backend.sh
./run_frontend.sh
./model_api/run_model_api.sh
```

프론트엔드는 <http://localhost:8501>, API 문서는 <http://localhost:8000/docs>, 헬스 체크는 <http://localhost:8000/health>에서 확인할 수 있습니다. 임베딩 API는 <http://localhost:8001/health>에서 확인할 수 있습니다. `.env`는 프로젝트 루트에 두며 Git에서 제외됩니다.

임베딩 API는 첫 실행 시 `BAAI/bge-m3` 모델을 다운로드합니다(약 2.3GB). `EMBED_MODEL_NAME` 환경 변수로 모델 이름을 바꿀 수 있습니다.

## 설정

기본 모드에서는 외부 API 키가 필요하지 않습니다. `.env`에서 각 단계를 HTTP 또는 Raon 모드로 바꾸면 지정한 API에 연결됩니다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `EMBEDDER_MODE` | `mock` | `mock` 또는 `http` |
| `EMBEDDER_URL` | 빈 값 (예시 URL은 `.env.example` 참조) | 임베딩 API 전체 URL (`http` 모드에서 필수) |
| `EMBED_DIM` | `768` | mock 임베딩 벡터 차원 |
| `VECTOR_SEARCH_MODE` | `mock` | `mock` 또는 `pgvector` |
| `RERANKER_MODE` | `off` | `off`, `mock` 또는 `http` |
| `RERANKER_URL` | 빈 값 | 재정렬 API 전체 URL (`http` 모드에서 필수) |
| `RERANK_CANDIDATES` | `20` | 재정렬을 켰을 때 검색할 후보 수 |
| `GENERATOR_MODE` | `mock` | `mock`, `http` 또는 `raon` |
| `GENERATOR_URL` | 빈 값 | 생성 API 전체 URL (`http` 모드에서 필수) |
| `PROVIDER_TIMEOUT` | `10` | 임베딩/재정렬 외부 API 호출 타임아웃(초) |
| `GENERATOR_TIMEOUT` | `120` | 생성 API 타임아웃(초) |
| `RAON_API_KEY` | 빈 값 | Raon API 인증 키 (`GENERATOR_MODE=raon`에서 필수) |
| `RAON_BASE_URL` | `https://raon-api.krafton.ai/v1` | Raon API 기본 URL |
| `RAON_MODEL` | `raon-1-mini-preview` | Raon 모델 |
| `RAON_MAX_TOKENS` | `4096` | Raon 최대 생성 토큰 |
| `RAON_TEMPERATURE` | `1.0` | Raon 생성 temperature |
| `DEBUG` | `false` | `true`이면 생성기의 `thinking`을 응답에 포함 |
| `BACKEND_URL` | `http://localhost:8000` | Streamlit이 호출할 백엔드 URL. EC2에서는 `http://127.0.0.1:8000` |

각 단계의 HTTP API 오류/타임아웃은 실패 단계명이 포함된 HTTP 503 응답으로 전달됩니다. `PROVIDER_TIMEOUT`은 임베딩/재정렬에, `GENERATOR_TIMEOUT`은 생성에 적용됩니다.
Raon 직접 호출(`GENERATOR_MODE=raon`)은 로컬 테스트용이며, 배포 시에는 ai-gateway(`http` 모드)를 사용합니다.

## 백엔드 API 명세

### `GET /health`

```json
{
  "status": "ok",
  "stages": {
    "embedder": "mock",
    "vector_search": "mock",
    "reranker": "off",
    "generator": "mock"
  }
}
```

### `POST /chat`

요청 (`top_k`는 선택, 기본값 5, 허용 범위 1–50):

```json
{"question": "Where is the Eiffel Tower?", "top_k": 5}
```

응답:

```json
{
  "answer": "The Eiffel Tower is in Paris, France.",
  "no_answer": false,
  "sources": [
    {"doc_id": "doc-123", "title": "Eiffel Tower", "text": "The Eiffel Tower is in Paris.", "score": 0.92}
  ],
  "latency_ms": {
    "embed": 3, "search": 12, "rerank": 0, "generation": 83, "total": 98
  },
  "model": "raon-1-mini"
}
```

`title`이 없으면 `null`입니다. `DEBUG=true`인 경우에만 응답에 `thinking` 문자열 필드가 추가됩니다. 추론 과정은 `answer`에 포함하지 않아야 합니다. 근거가 검색되지 않으면 생성 단계를 건너뛰고 `no_answer: true`, 빈 `sources`, `generation: 0`을 반환합니다.

외부 API 실패 응답 예 (HTTP 503):

```json
{
  "error": {
    "stage": "embed",
    "message": "Embed service failed: ..."
  }
}
```

`stage`는 `embed`, `search`, `rerank`, `generation` 중 하나입니다.

## 팀원 API 계약

백엔드는 필요한 단계만 팀원 API에 JSON POST 요청을 보냅니다. 각 URL은 단계별 환경 변수에 설정합니다.

### 임베딩 API (`EMBEDDER_URL`)
요청:

```json
{"texts": ["에펠탑은 어디에 있나요?"]}
```

성공 응답:

```json
{
  "embeddings": [[0.12, -0.08, 0.34]],
  "model": "multilingual-embedder",
  "dim": 3
}
```

`embeddings`는 요청한 텍스트 수와 같은 길이의 벡터 배열이며, 각 벡터의 길이는 `dim`과 같아야 합니다.

`model_api`는 이 계약에 따라 `POST /embed`와 `GET /health`를 제공합니다. 임베딩 벡터는 코사인 거리 검색을 위해 정규화됩니다.

### 재정렬 API (`RERANKER_URL`)
요청:

```json
{
  "question": "에펠탑은 어디에 있나요?",
  "candidates": [
    {"doc_id": "doc-123", "text": "에펠탑은 프랑스 파리에 있습니다."}
  ]
}
```

성공 응답:

```json
{
  "results": [
    {"doc_id": "doc-123", "score": 0.92}
  ]
}
```

결과는 점수가 높은 순서로 반환합니다. 백엔드는 이 순서대로 후보를 재정렬하고 점수를 덮어씁니다. `RERANKER_MODE=off`이면 이 단계를 건너뜁니다.

### 답변 생성 API (`GENERATOR_URL`)
요청:

```json
{
  "question": "에펠탑은 어디에 있나요?",
  "context": [
    {
      "doc_id": "doc-123",
      "title": "에펠탑",
      "text": "에펠탑은 프랑스 파리에 있습니다.",
      "score": 0.92
    }
  ]
}
```

성공 응답:

```json
{
  "answer": "에펠탑은 프랑스 파리에 있습니다.",
  "no_answer": false,
  "model": "raon-1-mini",
  "thinking": "Optional internal reasoning, omitted unless DEBUG=true."
}
```

`answer`, `no_answer`, `model`은 필수입니다. `thinking`은 선택 사항이며 사용자에게 노출할 최종 답변이 아닙니다. 답을 뒷받침할 수 없으면 `no_answer: true`로 설정하고, `answer`에는 사용자에게 보여줄 안내 문구를 넣습니다.

Raon 직접 호출을 확인하려면 `.env`에 `RAON_API_KEY`를 설정하고 backend 디렉터리에서 실행합니다:

```bash
python -m scripts.raon_smoke
```

## EC2 배포 (Ubuntu)

아래 예시는 저장소를 `/home/ubuntu/RAG_01`에 배치합니다.

### 1. Python 3.11 설치 및 코드 받기

Ubuntu 22.04/24.04에서 Python 3.11을 설치합니다.

```bash
sudo apt update
sudo apt install -y software-properties-common git
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv
git clone <REPOSITORY_URL> /home/ubuntu/RAG_01
cd /home/ubuntu/RAG_01
cp .env.example .env
```

`.env`에서 단계별 모드와 팀원 API URL을 설정합니다. 비밀 값은 저장소에 커밋하지 않습니다.

### 2. 가상환경과 의존성 설치

```bash
cd /home/ubuntu/RAG_01
python3.11 -m venv backend/.venv
python3.11 -m venv frontend/.venv
python3.11 -m venv model_api/.venv
backend/.venv/bin/python -m pip install -r backend/requirements.txt
frontend/.venv/bin/python -m pip install -r frontend/requirements.txt
model_api/.venv/bin/python -m pip install -r model_api/requirements.txt
```

### 3. systemd 서비스 등록 및 시작

서비스 예제의 경로는 `/home/ubuntu/RAG_01`과 `ubuntu` 사용자를 기준으로 합니다. 저장소가 다른 경로라면 서비스 파일의 `WorkingDirectory`, `EnvironmentFile`, `ExecStart`를 수정합니다.

```bash
sudo cp deploy/backend.service /etc/systemd/system/rag-backend.service
sudo cp deploy/frontend.service /etc/systemd/system/rag-frontend.service
sudo cp deploy/model-api.service /etc/systemd/system/rag-model-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now rag-backend.service rag-frontend.service rag-model-api.service
sudo systemctl status rag-backend.service rag-frontend.service rag-model-api.service
```

로그 확인과 재시작:

```bash
sudo journalctl -u rag-backend.service -f
sudo journalctl -u rag-frontend.service -f
sudo journalctl -u rag-model-api.service -f
sudo systemctl restart rag-backend.service rag-frontend.service rag-model-api.service
```

`enable` 설정으로 EC2 재부팅 후 자동 시작하며, systemd가 SSH 연결과 독립적으로 세 프로세스를 계속 실행하고 실패 시 재시작합니다.

### 4. EC2 보안 그룹

인바운드 규칙은 다음처럼 설정합니다.

| 포트 | 공개 범위 | 용도 |
|---|---|---|
| 22/TCP | 관리자 IP만 허용 | SSH 관리 |
| 8501/TCP | 사용자 접근이 필요한 주소 또는 대역 | Streamlit 웹 화면 |
| 8000/TCP | **외부 인바운드 규칙을 만들지 않음** | 백엔드 API. Streamlit과 같은 EC2 내 `127.0.0.1`로만 접근 |
| 8001/TCP | **외부 인바운드 규칙을 만들지 않음** | 임베딩 API. 백엔드와 같은 EC2 내 `127.0.0.1`로만 접근 |

백엔드 서비스는 loopback 주소(`127.0.0.1:8000`)에만 바인딩되며, 프론트엔드는 내부 주소 `http://127.0.0.1:8000`을 사용합니다. 따라서 백엔드 포트를 인터넷에 노출하지 않습니다.
임베딩 API도 `127.0.0.1:8001`에만 바인딩되므로 외부에 노출하지 않습니다.

## 테스트

```bash
cd backend
python -m pytest
```
