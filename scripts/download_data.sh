#!/usr/bin/env bash
# Adult Census Income 데이터셋 다운로드
# data/raw/ 는 git에 커밋하지 않으므로, 팀원은 이 스크립트로 데이터를 재현한다.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data/raw
BASE_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/adult"

curl -sSL -o data/raw/adult.data "${BASE_URL}/adult.data"
curl -sSL -o data/raw/adult.names "${BASE_URL}/adult.names"
curl -sSL -o data/raw/adult.test "${BASE_URL}/adult.test"

echo "다운로드 완료:"
wc -l data/raw/adult.data
