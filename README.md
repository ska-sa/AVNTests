
# AVNTests
AVN Correlator Unit and Acceptance Testing based Framework

## Installation

### Python Core packages

Install dependencies to the system, by following their installation instructions:

* [_katcp-python_](https://github.com/ska-sa/katcp-python)
* [_nosekatreport_](https://github.com/ska-sa/nosekatreport/tree/karoocbf)

### Python testing dependencies packages

It is highly recommended to install dependencies on a [_Python virtual environment_](https://virtualenv.pypa.io/).

#### Setup Virtualenv
Step-by-step instructions.
```
# Install Python essentials and pip
$ curl -s https://bootstrap.pypa.io/get-pip.py | python
$ pip install --user virtualenv

# Install testing dependencies in the virtualenv
$ git clone https://github.com/ska-sa/AVNTests
$ cd AVNTests
$ virtualenv .venv
$ . .venv/bin/activate
$ $(which pip) install -e .
```

## Unit Testing

Running unit-testing.
```
$ . venv/bin/activate
# to follow...
```

## Contributors

 * Mpho Mphego
