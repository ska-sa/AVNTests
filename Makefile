.PHONY: channelisation linearity gain accumulation clean bootstrap venv tests

channelisation:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=INFO avn_tests/test_avn.py:test_AVN.test_channelisation'
linearity:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=INFO avn_tests/test_avn.py:test_AVN.test_linearity'
gain:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=DEBUG avn_tests/test_avn.py:test_AVN.test_digital_gain'
accumulation:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=DEBUG avn_tests/test_avn.py:test_AVN.test_accumulation_length'
tests:
	sh -c '. .venv/bin/activate; nosetests -sv --with-katreport --logging-level=DEBUG avn_tests/test_avn.py:test_AVN'
clean:
	rm -rf build dist .venv || true;

bootstrap: venv
	.venv/bin/pip install -r pip-requirements.txt
	.venv/bin/python setup.py install -f

venv:
	virtualenv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
	
