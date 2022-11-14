MAX_LINE_LENGTH=120
TARGET_PYTHON_VERSION=py38

all: style lint type

style:
	# Check only
	poetry run black --target-version ${TARGET_PYTHON_VERSION} \
	--check \
	--verbose \
	--line-length ${MAX_LINE_LENGTH} \
	src/ tests/

format:
	# Format all files
	poetry run black --target-version ${TARGET_PYTHON_VERSION} \
	--verbose \
	--line-length ${MAX_LINE_LENGTH} \
	src/ tests/

type: 
	poetry run mypy src/ tests/ \
	--ignore-missing-imports \
	--follow-imports=skip \
	--show-error-context

lint:
	poetry run pylint src/ tests/ \
	--disable=C,W,no-error,design,no-member,duplicate-code,unnecessary-comprehension,import-error,no-name-in-module \
	--max-line-length=${MAX_LINE_LENGTH} \
	--enable=C0303,W0613,R1705,C0303,C0305,W0601,W0641,W0611,W0613,W0614,W0631,W0602
