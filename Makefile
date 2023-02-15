test:
	nose2 -v

hint:
	pytype nttw

test-hint: test hint
	echo 'Finished running tests and checking type hints'

lint:
	pylint nttw
	pycodestyle nttw/*.py
	pycodestyle nttw/**/*.py
