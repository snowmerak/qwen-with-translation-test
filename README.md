# Qwen with translation test

한국어 요청을 Hy-MT2로 영어로 번역한 뒤 Qwen에 전달하고, Qwen의 영어
응답을 다시 Hy-MT2로 한국어로 번역하는 작은 비교 실험용 CLI입니다.

Qwen에는 영어 대화 이력만 전달합니다. SQLite에는 원문과 번역문을 함께
저장하므로 실제로 어떤 번역이 모델에 입력됐는지도 나중에 확인할 수 있습니다.

## 처리 흐름

```text
한국어 입력
  -> Hy-MT2 (Korean -> English)
  -> Qwen (영어 컨텍스트로 응답)
  -> Hy-MT2 (English -> Korean)
  -> 최종 한국어 응답 출력
```

Hy-MT2 호출에는 Tencent가 공개한 기본 번역 지시문을 사용합니다. 별도의
시스템 프롬프트는 넣지 않습니다.

## 설치

```powershell
Copy-Item .env.example .env
uv sync
```

기본 설정은 다음과 같습니다.

```dotenv
OPENAI_BASE_URL=http://macstudio:11888/v1
OPENAI_API_KEY=local
HY_MODEL=Hy-MT2-30B-A3B-MLX-4bit
QWEN_MODEL=Qwen3.8-27B-4bit
DATABASE_PATH=chat.db
```

로컬 OpenAI 호환 서버가 인증을 검사하지 않더라도 OpenAI Python SDK에는
비어 있지 않은 API 키가 필요하므로 기본값으로 `local`을 사용합니다.

## 실행

대화형으로 시작하면 매번 새 conversation이 만들어집니다.

```powershell
uv run qwen-translate-chat
```

한 번만 요청하려면 다음과 같이 실행합니다. 이 모드의 표준 출력에는 최종
한국어 응답만 표시됩니다.

```powershell
uv run qwen-translate-chat --once "Python의 장점을 세 가지 알려줘"
```

중간 영어 번역도 함께 확인할 수 있습니다.

```powershell
uv run qwen-translate-chat --once "Python이 뭐야?" --show-english
```

저장된 대화를 확인하거나 이어서 대화할 수 있습니다.

```powershell
uv run qwen-translate-chat --list-conversations
uv run qwen-translate-chat --conversation 1
```

서버가 노출하는 실제 모델 ID 확인:

```powershell
uv run qwen-translate-chat --list-models
```

대화형 명령은 `/history`, `/quit` 두 가지입니다.

## SQLite 구조

`messages` 테이블의 각 행에는 아래 값이 함께 저장됩니다.

- `role`: `user` 또는 `assistant`
- `content_en`: Qwen 컨텍스트에 사용하는 영어 메시지
- `content_ko`: 사용자 원문 또는 최종 한국어 번역

테스트 실행:

```powershell
uv run pytest
```

## Hy 모델 로드 문제

이 MLX 변환본에는 `hy_v3.py` 커스텀 모델 코드가 포함되어 있습니다. oMLX에서
아래와 같은 409 오류가 나오면 `http://macstudio:11888/admin`의 Hy 모델 설정에서
**Trust Remote Code**를 활성화한 뒤 모델을 다시 로드해야 합니다.

```text
requires executing custom model code ('hy_v3.py').
Pass trust_remote_code=True if you trust this model.
```

서버를 환경 변수로 실행하는 구성이라면 `OMLX_TRUST_REMOTE_CODE=true`를 사용할
수 있습니다. 커스텀 코드를 신뢰할 수 있는지 확인한 경우에만 활성화하세요.
