# Qwen with translation test

한국어 요청을 Hy-MT2로 영어, 중국어, 한국어 또는 일본어로 번역한 뒤 Qwen에
전달하고, Qwen 응답을 다시 Hy-MT2로 한국어로 번역하는 비교 실험용 CLI입니다.
번역 효과를 비교할 수 있도록 Hy-MT2를 완전히 생략하는 별도 bypass 모드도
지원합니다.

Qwen에는 선택한 피벗 언어의 대화 이력만 전달합니다. SQLite에는 원문과
번역문을 함께 저장하므로 실제 입력도 나중에 확인할 수 있습니다. 한 conversation
안에서는 같은 피벗 언어를 계속 사용합니다.

## 처리 흐름

```text
한국어 입력
  -> Hy-MT2 (Korean -> English, Chinese, Korean 또는 Japanese)
  -> Qwen (선택한 피벗 언어 컨텍스트로 응답)
  -> Hy-MT2 (피벗 언어 -> Korean)
  -> 최종 한국어 응답 출력
```

`Korean` 피벗도 Hy-MT2의 `Korean -> Korean` 번역과 `Korean -> Korean`
역번역을 모두 거칩니다. 번역 호출 없이 한국어를 Qwen에 직접 전달하려면
피벗 언어가 아니라 `--bypass-translation` 옵션을 사용합니다.

Hy-MT2 호출에는 Tencent가 공개한 기본 번역 지시문을 사용합니다. 별도의
시스템 프롬프트는 넣지 않습니다. 영어, 중국어, 한국어, 일본어 피벗은 각각
해당 언어로 작성된 번역 지시문을 양방향 번역에 사용합니다. 예를 들어 중국어
지시문은 다음과 같습니다.

```text
将以下文本翻译为中文，注意只需要输出翻译后的结果，不要额外解释：

{source_text}
```

한국어로 되돌릴 때는 대상 언어만 `韩语`로 바뀝니다.

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
PIVOT_LANGUAGE=English
```

로컬 OpenAI 호환 서버가 인증을 검사하지 않더라도 OpenAI Python SDK에는
비어 있지 않은 API 키가 필요하므로 기본값으로 `local`을 사용합니다.

## 실행

대화형으로 시작하면 매번 새 conversation이 만들어집니다.

```powershell
uv run qwen-translate-chat
```

피벗 언어를 선택하려면 다음 옵션을 사용합니다. `한국어`, `일어`, `ko`, `ja`
같은 별칭도 사용할 수 있습니다.

```powershell
uv run qwen-translate-chat --pivot-language chinese
uv run qwen-translate-chat --pivot-language korean
uv run qwen-translate-chat --pivot-language japanese
```

Hy-MT2를 호출하지 않고 한국어 입출력을 Qwen에 직접 전달하는 새 대화를
시작하려면 별도 bypass 옵션을 사용합니다. 이 모드는 대화에 저장되므로 해당
conversation을 재개해도 계속 유지됩니다.

```powershell
uv run qwen-translate-chat --bypass-translation
# 짧은 별칭
uv run qwen-translate-chat --bypass
```

한 번만 요청하려면 다음과 같이 실행합니다. 이 모드의 표준 출력에는 최종
한국어 응답만 표시됩니다.

```powershell
uv run qwen-translate-chat --once "Python의 장점을 세 가지 알려줘"
```

중간 피벗 언어 번역도 함께 확인할 수 있습니다.

```powershell
uv run qwen-translate-chat --once "Python이 뭐야?" --show-pivot
uv run qwen-translate-chat --pivot-language chinese --once "Python이 뭐야?" --show-pivot
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
- `content_pivot`: Qwen 컨텍스트에 사용하는 피벗 언어 메시지
- `content_ko`: 사용자 원문 또는 최종 한국어 번역

`conversations.translation_bypass`에는 해당 대화가 Hy-MT2를 생략하는지
저장됩니다. bypass 대화에서는 `content_pivot`과 `content_ko`가 같은 한국어
내용입니다.

기존 `content_en` 기반 DB는 처음 열 때 `content_pivot` 구조로 자동
마이그레이션되며 기존 conversation은 영어 피벗으로 유지됩니다.

테스트 실행:

```powershell
uv run pytest
```

## 피벗 및 bypass 벤치마크

10개 한국어 케이스를 각각 4회 반복하며, 기본적으로 매 반복에서 영어, 중국어,
한국어, 일본어 피벗을 모두 실행합니다. 결과는 40개의 비교 레코드로 저장되며
각 레코드에는 네 피벗의 중간 번역, Qwen 원문, 최종 한국어와 단계별 소요 시간이
함께 들어갑니다.

```powershell
uv run qwen-translate-benchmark
```

비교할 피벗을 제한하거나 번역 bypass 결과까지 포함할 수 있습니다.

```powershell
uv run qwen-translate-benchmark `
  --pivot-languages English Chinese Korean Japanese `
  --include-bypass
```

기본 결과 파일은 `results/pivot_benchmark.jsonl`, 대화 원본은
`benchmark.db`에 저장됩니다. 사람이 나란히 평가하기 위한
`results/pivot_benchmark.csv`도 함께 생성되며, 각 비교 모드의 품질 점수와 선호
모드, 평가 메모 열은 비워 둡니다. 실행이 중단되면 같은 비교 모드 조합으로
완료되지 않은 레코드부터 이어서 실행합니다. 출력 한도는 `.env`의
`HY_MAX_TOKENS`와 `QWEN_MAX_TOKENS`를 따르며, 기본값은 각각 4,096과
32,768입니다. thinking
토큰을 충분히 허용하기 위해 Qwen 한도를 더 크게 둡니다. 한 레코드만 시험하려면
다음처럼 실행합니다.

```powershell
uv run qwen-translate-benchmark --limit 1
```

실험별로 출력 한도를 바꾸려면 `--hy-max-tokens`와 `--qwen-max-tokens`를
각각 사용할 수 있습니다.

## 샌드박스 Python 도구

Qwen의 OpenAI-compatible function calling을 통해 `execute_python` 도구를
사용할 수 있습니다. Docker 대신 Rust로 작성된 제한형 Python 인터프리터인
[Pydantic Monty](https://github.com/pydantic/monty)를 별도 워커 프로세스로
실행합니다.

`.env`에서 명시적으로 활성화합니다.

```dotenv
PYTHON_TOOL_ENABLED=true
PYTHON_SANDBOX_TIMEOUT=5
PYTHON_SANDBOX_MAX_MEMORY_BYTES=33554432
PYTHON_SANDBOX_MAX_RECURSION=100
PYTHON_SANDBOX_MAX_CODE_CHARS=20000
PYTHON_SANDBOX_MAX_OUTPUT_BYTES=20000
PYTHON_TOOL_MAX_ROUNDS=4
```

예시:

```powershell
uv run qwen-translate-chat --once `
  "파이썬 도구를 사용해서 1부터 100까지 정수의 제곱합을 계산해줘"
```

실행 코드에는 호스트 파일시스템, 환경변수, 네트워크 또는 외부 함수 접근 권한을
제공하지 않습니다. 실행 내역과 코드 및 결과는 SQLite의 `tool_executions`
테이블에 기록됩니다.

Monty는 일반 CPython 전체가 아니라 AI 생성 코드 실행에 필요한 Python
부분집합입니다. 기본 계산, 컬렉션, 문자열, JSON, 정규식, 날짜 처리 등은
가능하지만 NumPy/Pandas 같은 제3자 패키지와 지원되지 않는 표준 라이브러리는
사용할 수 없습니다. 현재 Monty 프로젝트 자체도 아직 experimental로 표시돼
있으므로 신뢰 경계를 넓히는 외부 함수나 디렉터리 mount는 추가하지 않는 것을
권장합니다.

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
