channelisation:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=INFO avn_tests/test_avn.py:test_AVN.test_channelisation'

clean:
	rm -rf build dist
	rm -rf .venv

bootstrap: venv
	.venv/bin/pip install -r pip-requirements.txt
	.venv/bin/python setup.py install -f

venv:
	virtualenv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install --upgrade setuptools